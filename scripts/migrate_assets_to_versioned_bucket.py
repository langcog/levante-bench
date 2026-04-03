#!/usr/bin/env python3
"""
Copy LEVANTE assets from source bucket layout into a versioned destination.

Source layout (public):
  corpus/<task>/<file>.csv
  visual/<task>/...
  manifest.csv
  translations/item-bank-translations.csv

Destination layout:
  gs://<dest-bucket>/<dest-root-prefix>/<version>/corpus/<task>/<file>.csv
  gs://<dest-bucket>/<dest-root-prefix>/<version>/visual/<task>/...
  gs://<dest-bucket>/<dest-root-prefix>/<version>/manifest.csv
  gs://<dest-bucket>/<dest-root-prefix>/<version>/translations/item-bank-translations.csv
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_MAPPING = REPO_ROOT / "src" / "levante_bench" / "config" / "task_name_mapping.csv"
DEFAULT_SOURCE_BASE = "https://storage.googleapis.com/levante-assets-prod"
DEFAULT_DEST_ROOT_PREFIX = "corpus_data"


def _bucket_and_prefix_from_base(base: str) -> tuple[str, str]:
    """Parse bucket name and optional path-prefix from source base URL."""
    parsed = urlparse(base.rstrip("/"))
    host = parsed.netloc
    parts = [p for p in parsed.path.strip("/").split("/") if p]

    if host == "storage.googleapis.com":
        if not parts:
            raise ValueError(f"Invalid source base URL: {base}")
        return parts[0], "/".join(parts[1:])

    if host.endswith(".storage.googleapis.com"):
        bucket = host.split(".storage.googleapis.com")[0]
        return bucket, "/".join(parts)

    if parts:
        return parts[0], "/".join(parts[1:])
    raise ValueError(f"Could not parse source bucket from URL: {base}")


def _list_bucket_keys(bucket_name: str, prefix: str) -> list[str]:
    url = f"https://{bucket_name}.storage.googleapis.com"
    ns = "http://doc.s3.amazonaws.com/2006-03-01"
    params: dict[str, str] = {"prefix": prefix, "list-type": "2"}
    keys: list[str] = []
    while True:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall(f".//{{{ns}}}Contents"):
            k = c.find(f"{{{ns}}}Key")
            if k is not None and k.text:
                keys.append(k.text)
        truncated = root.find(f".//{{{ns}}}IsTruncated")
        if truncated is not None and truncated.text == "true":
            next_token = root.find(f".//{{{ns}}}NextContinuationToken")
            if next_token is not None and next_token.text:
                params = {
                    "prefix": prefix,
                    "list-type": "2",
                    "continuation-token": next_token.text,
                }
            else:
                break
        else:
            break
    return keys


def _load_tasks(task_filter: str | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(TASK_MAPPING, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            internal = (row.get("internal_name") or "").strip()
            benchmark = (row.get("benchmark_name") or "").strip()
            corpus = (row.get("corpus_file") or "").strip()
            if not internal or not corpus:
                continue
            if task_filter and task_filter not in {internal, benchmark}:
                continue
            rows.append(
                {
                    "internal_name": internal,
                    "benchmark_name": benchmark,
                    "corpus_file": corpus,
                }
            )
    return rows


def _run_cp(src: str, dst: str, dry_run: bool) -> None:
    cmd = ["gcloud", "storage", "cp", src, dst]
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def _run_cp_recursive(src: str, dst: str, dry_run: bool) -> None:
    cmd = ["gcloud", "storage", "cp", "--recursive", src, dst]
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def _object_exists(bucket_name: str, key: str) -> bool:
    keys = _list_bucket_keys(bucket_name=bucket_name, prefix=key)
    return key in set(keys)


def run(
    version: str,
    dest_bucket: str,
    dest_root_prefix: str = DEFAULT_DEST_ROOT_PREFIX,
    source_base_url: str = DEFAULT_SOURCE_BASE,
    task_filter: str | None = None,
    dry_run: bool = False,
) -> None:
    tasks = _load_tasks(task_filter=task_filter)
    if not tasks:
        raise RuntimeError("No tasks selected for migration.")

    source_bucket, source_root_prefix = _bucket_and_prefix_from_base(source_base_url)
    dest_bucket = dest_bucket.replace("gs://", "").strip("/")
    dest_root_prefix = dest_root_prefix.strip("/")
    if dest_root_prefix:
        dest_prefix = f"gs://{dest_bucket}/{dest_root_prefix}/{version}"
    else:
        dest_prefix = f"gs://{dest_bucket}/{version}"

    source_prefix = source_root_prefix.strip("/")
    source_root = f"gs://{source_bucket}/{source_prefix}" if source_prefix else f"gs://{source_bucket}"
    # Optional manifest snapshot.
    source_manifest_key = f"{source_prefix}/manifest.csv" if source_prefix else "manifest.csv"
    if _object_exists(source_bucket, source_manifest_key):
        _run_cp(f"{source_root}/manifest.csv", f"{dest_prefix}/manifest.csv", dry_run=dry_run)
    else:
        print("Source manifest.csv not found at bucket root; skipping manifest copy.")

    # Optional translations snapshot.
    source_translations_key = (
        f"{source_prefix}/translations/item-bank-translations.csv"
        if source_prefix
        else "translations/item-bank-translations.csv"
    )
    if _object_exists(source_bucket, source_translations_key):
        _run_cp(
            f"{source_root}/translations/item-bank-translations.csv",
            f"{dest_prefix}/translations/item-bank-translations.csv",
            dry_run=dry_run,
        )
    else:
        print(
            "Source translations/item-bank-translations.csv not found; "
            "skipping translations copy."
        )

    op_count = 0
    for t in tasks:
        internal = t["internal_name"]
        corpus_file = t["corpus_file"]
        corpus_key = f"corpus/{internal}/{corpus_file}"
        source_corpus_key = f"{source_prefix}/{corpus_key}" if source_prefix else corpus_key
        _run_cp(
            f"gs://{source_bucket}/{source_corpus_key}",
            f"{dest_prefix}/{corpus_key}",
            dry_run=dry_run,
        )
        op_count += 1

        source_visual_prefix = (
            f"{source_prefix}/visual/{internal}" if source_prefix else f"visual/{internal}"
        )
        _run_cp_recursive(
            f"gs://{source_bucket}/{source_visual_prefix}",
            f"{dest_prefix}/visual/{internal}",
            dry_run=dry_run,
        )
        op_count += 1

    print(f"Completed {op_count} copy operations")

    if not dry_run:
        print(
            f"Migration complete. Assets copied to version prefix: "
            f"{dest_prefix}/"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Migrate LEVANTE assets into a versioned destination bucket prefix."
    )
    p.add_argument("--version", required=True, help="Destination version (YYYY-MM-DD).")
    p.add_argument(
        "--dest-bucket",
        required=True,
        help="Destination bucket name or gs:// URI (e.g. gs://levante-bench).",
    )
    p.add_argument(
        "--dest-root-prefix",
        default=DEFAULT_DEST_ROOT_PREFIX,
        help=(
            "Destination folder/prefix in bucket before <version>. "
            "Default: corpus_data (set '' for bucket root)."
        ),
    )
    p.add_argument(
        "--source-base-url",
        default=DEFAULT_SOURCE_BASE,
        help="Source public bucket base URL (default: levante-assets-prod).",
    )
    p.add_argument(
        "--task",
        default=None,
        help="Optional task filter (internal_name or benchmark_name).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print copy commands only.")
    args = p.parse_args()
    run(
        version=args.version,
        dest_bucket=args.dest_bucket,
        dest_root_prefix=args.dest_root_prefix,
        source_base_url=args.source_base_url,
        task_filter=args.task,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
