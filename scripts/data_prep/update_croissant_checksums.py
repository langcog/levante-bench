#!/usr/bin/env python3
"""Update sha256 values in a Croissant JSON-LD file.

This script recalculates checksums for distribution file entries and writes
them back into the Croissant metadata. It is designed around LEVANTE's current
distribution URL patterns:

- assets hosted under https://storage.googleapis.com/levante-bench/corpus_data/
- responses referenced with raw GitHub URLs under /main/data/...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_local_path(content_url: str, repo_root: Path) -> Path | None:
    parsed = urlparse(content_url)

    # Handle absolute local path-like entries.
    if not parsed.scheme:
        candidate = (repo_root / content_url).resolve()
        return candidate if candidate.exists() else None

    # Handle raw GitHub URLs (used for response manifests).
    gh_prefix = "https://raw.githubusercontent.com/langcog/levante-bench/main/"
    if content_url.startswith(gh_prefix):
        rel = content_url[len(gh_prefix) :]
        candidate = (repo_root / rel).resolve()
        return candidate if candidate.exists() else None

    # Handle assets bucket URLs.
    bucket_prefix = "https://storage.googleapis.com/levante-bench/corpus_data/"
    if content_url.startswith(bucket_prefix):
        rel = content_url[len(bucket_prefix) :]
        # Most files map under data/assets/<version>/...
        primary = (repo_root / "data" / "assets" / rel).resolve()
        if primary.exists():
            return primary
        # Current assets manifest is kept at data/assets/manifest.csv.
        if rel == "v1/manifest.csv":
            fallback = (repo_root / "data" / "assets" / "manifest.csv").resolve()
            if fallback.exists():
                return fallback
        return None

    # Handle responses manifest bucket URLs.
    responses_prefix = "https://storage.googleapis.com/levante-bench/responses_manifests/"
    if content_url.startswith(responses_prefix):
        rel = content_url[len(responses_prefix) :]
        candidate = (repo_root / "data" / "responses" / rel).resolve()
        return candidate if candidate.exists() else None

    return None


def update_checksums(*, croissant_path: Path, repo_root: Path, check_only: bool) -> int:
    data = json.loads(croissant_path.read_text(encoding="utf-8"))
    dist = data.get("distribution", [])
    if not isinstance(dist, list):
        raise RuntimeError("Croissant metadata has non-list 'distribution'.")

    changed = 0
    inspected = 0
    failures: list[str] = []
    for entry in dist:
        if not isinstance(entry, dict):
            continue
        content_url = str(entry.get("contentUrl", "")).strip()
        if not content_url:
            continue
        inspected += 1
        local_path = _resolve_local_path(content_url=content_url, repo_root=repo_root)
        if local_path is None:
            failures.append(f"Could not resolve local file for URL: {content_url}")
            continue
        digest = _sha256(local_path)
        old = str(entry.get("sha256", "")).strip()
        if old != digest:
            entry["sha256"] = digest
            changed += 1
            print(f"updated: {entry.get('@id', '<unknown>')} -> {digest}")
        else:
            print(f"unchanged: {entry.get('@id', '<unknown>')} -> {digest}")

    if failures:
        for msg in failures:
            print(f"ERROR: {msg}")
        return 2

    print(f"inspected distributions: {inspected}")
    print(f"changed checksums: {changed}")

    if not check_only and changed > 0:
        croissant_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote: {croissant_path}")

    return 1 if check_only and changed > 0 else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Update Croissant distribution checksums.")
    parser.add_argument(
        "--croissant-json",
        default="datasets/v1/levante_v1.croissant.json",
        help="Path to Croissant JSON-LD file.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root path used to resolve local files.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check only (do not write file). Exits 1 if updates are needed.",
    )
    args = parser.parse_args()

    return update_checksums(
        croissant_path=Path(args.croissant_json),
        repo_root=Path(args.repo_root).resolve(),
        check_only=bool(args.check),
    )


if __name__ == "__main__":
    raise SystemExit(main())
