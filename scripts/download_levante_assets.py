#!/usr/bin/env python3
"""
Download LEVANTE corpus and visual assets from the public GCP bucket.
Writes to data/assets/<version>/ and builds an item_uid -> local paths index.
Version defaults to today (YYYY-MM-DD). Idempotent.
"""

import argparse
import csv
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

# Retry transient connection errors
DOWNLOAD_RETRIES = 4
DOWNLOAD_RETRY_BACKOFF = 2.0  # seconds, doubled each retry

# Resolve package from script: repo root is parent of scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _project_root() -> Path:
    return _REPO_ROOT


def _add_src_to_path() -> None:
    src = _REPO_ROOT / "src"
    if src.exists() and str(src) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(src))


_add_src_to_path()

try:
    # Import defaults from package when available.
    from levante_bench.config.defaults import get_assets_base_url, get_task_mapping_path  # noqa: E402
except ModuleNotFoundError:
    # Fallback for lean branches where levante_bench.config package side-imports missing modules.
    def get_task_mapping_path() -> Path:
        return _REPO_ROOT / "src" / "levante_bench" / "config" / "task_name_mapping.csv"

    def get_assets_base_url() -> str:
        return os.environ.get("LEVANTE_ASSETS_BUCKET_URL", "https://storage.googleapis.com/levante-assets-prod")


def _bucket_name_from_base(base: str) -> str:
    """e.g. https://storage.googleapis.com/levante-assets-prod -> levante-assets-prod"""
    return base.rstrip("/").split("/")[-1]


def _list_bucket_keys(bucket_name: str, prefix: str) -> list[str]:
    """List object keys under prefix via GCP XML API (public bucket)."""
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
                params = {"prefix": prefix, "list-type": "2", "continuation-token": next_token.text}
            else:
                break
        else:
            break
    return keys


def _download_file(base_url: str, key: str, dest: Path) -> None:
    url = f"{base_url.rstrip('/')}/{key}"
    parent = dest.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        if not parent.is_dir():
            raise
    last_err = None
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            r = requests.get(url, timeout=90, stream=True)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            return
        except (requests.exceptions.ConnectionError, ConnectionResetError, TimeoutError) as e:
            last_err = e
            if attempt < DOWNLOAD_RETRIES - 1:
                time.sleep(DOWNLOAD_RETRY_BACKOFF * (2**attempt))
            continue
    raise last_err


def _load_task_mapping(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            internal = (row.get("internal_name") or "").strip()
            corpus = (row.get("corpus_file") or "").strip()
            if internal and corpus:
                rows.append(
                    {
                        "benchmark_name": (row.get("benchmark_name") or "").strip(),
                        "internal_name": internal,
                        "corpus_file": corpus,
                    }
                )
    return rows


def _looks_like_path(value: str) -> bool:
    """True if value looks like a file path (has extension or path sep), not an option label."""
    s = str(value).strip()
    if not s:
        return False
    return "." in s or "/" in s


def _print_completeness(
    assets_dir: Path,
    tasks: list[dict],
    corpus_dir: Path,
    visual_dir: Path,
) -> None:
    """Report bucket files on disk and, where corpus has path-like refs, corpus-referenced assets."""
    print("\nCompleteness:")
    for t in tasks:
        internal_name = t["internal_name"]
        visual_local = visual_dir / internal_name
        # Bucket files: count what we have on disk (we don't re-list; show dir count)
        if not visual_local.exists():
            print(f"  {internal_name}: no visual/ dir (0 files) — bucket prefix visual/{internal_name}/ may be wrong or not yet downloaded")
        else:
            on_disk = list(visual_local.rglob("*"))
            files_on_disk = sum(1 for p in on_disk if p.is_file())
            print(f"  {internal_name}: {files_on_disk} files in visual/ (bucket files downloaded here)")
        # Corpus path-like refs: only values that look like paths (e.g. contain . or /)
        corpus_file = t["corpus_file"]
        corpus_path = corpus_dir / internal_name / corpus_file
        if not corpus_path.exists():
            continue
        df = pd.read_csv(corpus_path)
        asset_cols = [c for c in df.columns if re.match(r"image[0-9]+", c, re.I) or "path" in c.lower() or c in ("audio_file", "image")]
        seen: set[str] = set()
        for _, row in df.iterrows():
            for c in asset_cols:
                v = row.get(c)
                if pd.isna(v) or not v:
                    continue
                val = str(v).strip()
                if not _looks_like_path(val):
                    continue
                path = visual_local / val
                seen.add(str(path))
        if seen:
            found = sum(1 for p in seen if Path(p).exists())
            missing = len(seen) - found
            print(f"    -> corpus path-like refs: {found}/{len(seen)} on disk" + (f" ({missing} missing)" if missing else ""))


def run(
    version: str | None = None,
    task_filter: str | None = None,
    data_root: Path | None = None,
    base_url: str | None = None,
    check_completeness: bool = False,
) -> None:
    from datetime import datetime

    data_root = data_root or _project_root() / "data"
    base_url = base_url or get_assets_base_url()
    version = version or datetime.now().strftime("%Y-%m-%d")
    assets_dir = data_root / "assets" / version
    corpus_dir = assets_dir / "corpus"
    visual_dir = assets_dir / "visual"

    mapping_path = get_task_mapping_path()
    if not mapping_path.exists():
        raise FileNotFoundError(f"Task mapping not found: {mapping_path}")
    tasks = _load_task_mapping(mapping_path)
    if task_filter:
        tasks = [t for t in tasks if t["internal_name"] == task_filter or t["benchmark_name"] == task_filter]
    if not tasks:
        return

    bucket_name = _bucket_name_from_base(base_url)
    index: dict[str, dict] = {}  # item_uid -> { task, internal_name, corpus_row, image_paths }

    for t in tasks:
        internal_name = t["internal_name"]
        corpus_file = t["corpus_file"]
        # 1) Corpus CSV
        corpus_key = f"corpus/{internal_name}/{corpus_file}"
        out_corpus = corpus_dir / internal_name / corpus_file
        if not out_corpus.exists():
            _download_file(base_url, corpus_key, out_corpus)
        df = pd.read_csv(out_corpus)
        # 2) Visual assets under visual/{internal_name}/ (always download, even if corpus has no item_uid)
        prefix = f"visual/{internal_name}/"
        try:
            keys = _list_bucket_keys(bucket_name, prefix)
        except Exception:
            keys = []
        visual_local_dir = visual_dir / internal_name
        n_downloaded = 0
        n_skipped = 0
        for key in keys:
            rel = key[len(prefix) :] if key.startswith(prefix) else key
            rel = rel.rstrip("/")
            if not rel:
                continue  # skip directory placeholder keys (would create egma-math as a file)
            local_path = visual_local_dir / rel
            if not local_path.exists():
                _download_file(base_url, key, local_path)
                n_downloaded += 1
            else:
                n_skipped += 1
        print(f"  {internal_name}: {n_downloaded} downloaded, {n_skipped} already present ({len(keys)} keys in bucket)")
        # 3) Build index rows from corpus (only when corpus has item_uid)
        if "item_uid" not in df.columns:
            continue
        for _, row in df.iterrows():
            uid = row.get("item_uid")
            if pd.isna(uid):
                continue
            uid = str(uid).strip()
            corpus_row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            # Optional: derive image paths from corpus (e.g. columns that look like image paths)
            image_paths: list[str] = []
            for c in df.columns:
                if re.match(r"image[0-9]+", c, re.I) or "image" in c.lower() and "path" in c.lower():
                    v = row.get(c)
                    if not pd.isna(v) and v:
                        rel = str(v).strip()
                        local = visual_local_dir / rel
                        # Store path relative to assets_dir for portability
                        try:
                            image_paths.append(str(local.relative_to(assets_dir)))
                        except ValueError:
                            image_paths.append(str(local))
            index[uid] = {
                "task": t["benchmark_name"],
                "internal_name": internal_name,
                "corpus_row": corpus_row,
                "image_paths": image_paths,
            }

    index_path = assets_dir / "item_uid_index.json"
    # Convert non-JSON-serializable (e.g. numpy) in corpus_row
    def _serialize(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    index_serializable = {}
    for uid, v in index.items():
        cr = v.get("corpus_row") or {}
        paths = v.get("image_paths") or []
        index_serializable[uid] = {
            "task": v.get("task"),
            "internal_name": v.get("internal_name"),
            "corpus_row": {k: _serialize(cr[k]) for k in cr},
            "image_paths": paths,
        }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_serializable, f, indent=2)
    print(f"Wrote {index_path} ({len(index_serializable)} item_uids)")

    if check_completeness:
        _print_completeness(assets_dir, tasks, corpus_dir, visual_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="Download LEVANTE assets and build item_uid index.")
    p.add_argument("--version", default=None, help="Asset version (default: today YYYY-MM-DD)")
    p.add_argument("--task", default=None, help="Only download this task (internal_name or benchmark_name)")
    p.add_argument("--data-root", type=Path, default=None, help="Data root (default: project data/)")
    p.add_argument("--base-url", default=None, help="Bucket base URL (default: config)")
    p.add_argument("--check-completeness", action="store_true", help="At end, report corpus-referenced assets found vs missing")
    args = p.parse_args()
    run(version=args.version, task_filter=args.task, data_root=args.data_root, base_url=args.base_url, check_completeness=args.check_completeness)


if __name__ == "__main__":
    main()
