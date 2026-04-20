#!/usr/bin/env python3
"""Publish LEVANTE asset manifests (+ checksums) to GCS.

Uploads:
- data/assets/manifest.csv -> gs://<bucket>/<root>/<version>/manifest.csv
- data/assets/<version>/manifests/* -> gs://<bucket>/<root>/<version>/manifests/

Also writes checksums file:
- data/assets/<version>/manifests/CROISSANT-SHA256SUMS.txt
which is uploaded with the manifests directory.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from urllib.parse import urlparse


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_bucket_url(bucket_url: str) -> tuple[str, str]:
    """Return (bucket, root_prefix) from URL-like bucket path."""
    # Supports:
    # - https://storage.googleapis.com/levante-bench/corpus_data
    # - gs://levante-bench/corpus_data
    bucket_url = bucket_url.strip()
    if bucket_url.startswith("gs://"):
        tail = bucket_url[len("gs://") :]
        parts = [p for p in tail.split("/") if p]
        if not parts:
            raise ValueError(f"Invalid bucket URL: {bucket_url}")
        bucket = parts[0]
        prefix = "/".join(parts[1:])
        return bucket, prefix

    parsed = urlparse(bucket_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported bucket URL format: {bucket_url}")
    if parsed.netloc == "storage.googleapis.com":
        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            raise ValueError(f"Invalid storage URL path: {bucket_url}")
        bucket = parts[0]
        prefix = "/".join(parts[1:])
        return bucket, prefix
    if parsed.netloc.endswith(".storage.googleapis.com"):
        bucket = parsed.netloc.split(".storage.googleapis.com")[0]
        prefix = parsed.path.strip("/")
        return bucket, prefix
    raise ValueError(f"Unsupported storage host in URL: {bucket_url}")


def _run(cmd: list[str], dry_run: bool) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _write_checksums_file(
    *,
    manifest_csv: Path,
    manifests_dir: Path,
) -> Path:
    checksums_path = manifests_dir / "CROISSANT-SHA256SUMS.txt"
    lines: list[str] = []
    # record set used in Croissant distributions
    lines.append(f"{_sha256(manifest_csv)}  ../manifest.csv")
    for path in sorted(manifests_dir.glob("*")):
        if path.is_file() and path.name != checksums_path.name:
            lines.append(f"{_sha256(path)}  {path.name}")
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksums_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload assets manifest files and checksums to bucket."
    )
    parser.add_argument("--version", default="v1", help="Assets version label.")
    parser.add_argument(
        "--bucket-url",
        default="https://storage.googleapis.com/levante-bench/corpus_data",
        help="Bucket URL root used by asset downloader.",
    )
    parser.add_argument("--data-root", default="data", help="Local data root.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    data_root = (repo_root / args.data_root).resolve()
    manifest_csv = data_root / "assets" / "manifest.csv"
    manifests_dir = data_root / "assets" / str(args.version) / "manifests"

    if not manifest_csv.exists():
        raise RuntimeError(f"Missing local manifest CSV: {manifest_csv}")
    if not manifests_dir.exists():
        raise RuntimeError(f"Missing local manifests dir: {manifests_dir}")

    checksums_path = _write_checksums_file(
        manifest_csv=manifest_csv,
        manifests_dir=manifests_dir,
    )
    print(f"Wrote checksums: {checksums_path}")

    bucket, root_prefix = _parse_bucket_url(str(args.bucket_url))
    base = f"gs://{bucket}"
    if root_prefix:
        base = f"{base}/{root_prefix}"
    dest_version = f"{base}/{args.version}"
    dest_manifest = f"{dest_version}/manifest.csv"
    dest_manifests = f"{dest_version}/manifests"

    _run(
        ["gcloud", "storage", "cp", str(manifest_csv), dest_manifest],
        dry_run=bool(args.dry_run),
    )
    _run(
        ["gcloud", "storage", "rsync", "--recursive", str(manifests_dir), dest_manifests],
        dry_run=bool(args.dry_run),
    )

    print("Publish complete.")
    print(f"Manifest CSV: {dest_manifest}")
    print(f"Manifests dir: {dest_manifests}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
