#!/usr/bin/env python3
"""Compare Croissant distribution checksums against local files and remote URLs.

For each ``distribution`` entry with ``contentUrl`` + ``sha256`` in
``datasets/v1/levante_v1.croissant.json``:

- **embedded** — value recorded in the Croissant JSON
- **local** — SHA-256 of the file resolved the same way as
  ``update_croissant_checksums.py`` (repo-relative mirror under
  ``data/assets`` / ``data/responses``)
- **remote** — SHA-256 of the bytes at ``contentUrl`` (typically public GCS HTTPS)

Exit codes::

    0 — embedded matches both local and remote (all resolvable)
    1 — mismatch or missing local / remote fetch failure
    2 — could not resolve local path for one or more URLs

Examples::

    python scripts/data_prep/audit_croissant_sync.py
    python scripts/data_prep/audit_croissant_sync.py --skip-remote
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_url(url: str, *, timeout_s: int) -> str:
    hasher = hashlib.sha256()
    req = Request(url, headers={"User-Agent": "levante-bench-audit-croissant/1.0"})
    with urlopen(req, timeout=timeout_s) as resp:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_local_path(content_url: str, repo_root: Path) -> Path | None:
    """Mirror of ``update_croissant_checksums._resolve_local_path``."""
    parsed = urlparse(content_url)
    if not parsed.scheme:
        candidate = (repo_root / content_url).resolve()
        return candidate if candidate.exists() else None

    gh_prefix = "https://raw.githubusercontent.com/langcog/levante-bench/main/"
    if content_url.startswith(gh_prefix):
        rel = content_url[len(gh_prefix) :]
        candidate = (repo_root / rel).resolve()
        return candidate if candidate.exists() else None

    bucket_prefix = "https://storage.googleapis.com/levante-bench/corpus_data/"
    if content_url.startswith(bucket_prefix):
        rel = content_url[len(bucket_prefix) :]
        primary = (repo_root / "data" / "assets" / rel).resolve()
        if primary.exists():
            return primary
        if rel == "v1/manifest.csv":
            fallback = (repo_root / "data" / "assets" / "manifest.csv").resolve()
            if fallback.exists():
                return fallback
        return None

    responses_prefix = "https://storage.googleapis.com/levante-bench/responses_manifests/"
    if content_url.startswith(responses_prefix):
        rel = content_url[len(responses_prefix) :]
        candidate = (repo_root / "data" / "responses" / rel).resolve()
        return candidate if candidate.exists() else None

    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--croissant-json",
        type=Path,
        default=REPO_ROOT / "datasets" / "v1" / "levante_v1.croissant.json",
    )
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument(
        "--skip-remote",
        action="store_true",
        help="Only compare embedded vs local (no HTTP).",
    )
    p.add_argument("--timeout", type=int, default=120, help="Seconds per URL fetch.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    croissant_path = args.croissant_json.resolve()
    data = json.loads(croissant_path.read_text(encoding="utf-8"))
    dist = data.get("distribution", [])
    if not isinstance(dist, list):
        print("ERROR: invalid Croissant distribution list", file=sys.stderr)
        return 2

    unresolved = 0
    mismatches = 0

    for entry in dist:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("contentUrl", "")).strip()
        embedded = str(entry.get("sha256", "")).strip()
        eid = str(entry.get("@id", "?"))
        name = str(entry.get("name", ""))
        if not url or not embedded:
            continue

        local_path = resolve_local_path(url, repo)
        if local_path is None:
            print(f"[NO LOCAL] {eid} ({name})\n  url: {url}")
            unresolved += 1
            mismatches += 1
            continue

        local_hex = _sha256_file(local_path)
        ok_loc = local_hex == embedded

        if args.skip_remote:
            remote_hex = "(skipped)"
            ok_rem = True
        else:
            try:
                remote_hex = _sha256_url(url, timeout_s=args.timeout)
                ok_rem = remote_hex == embedded
            except Exception as exc:
                remote_hex = f"<fetch failed: {exc}>"
                ok_rem = False

        status = "OK" if (ok_loc and ok_rem) else "MISMATCH"
        if status != "OK":
            mismatches += 1

        print(f"\n[{status}] {eid} — {name}")
        print(f"  url:      {url}")
        print(f"  local:    {local_path.relative_to(repo)}")
        print(f"  embedded: {embedded}")
        print(f"  local_:   {local_hex}  {'MATCH' if ok_loc else 'DIFF'}")
        if not args.skip_remote:
            print(f"  remote_:  {remote_hex}  {'MATCH' if ok_rem else 'DIFF'}")

    print(
        f"\nSummary: unresolved_local={unresolved} "
        f"mismatch_or_error={mismatches} skip_remote={args.skip_remote}",
        flush=True,
    )
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
