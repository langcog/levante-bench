#!/usr/bin/env python3
"""Download benchmark results from a GCS bucket prefix into local results/."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_BUCKET_RESULTS_URL = "gs://levante-bench/results"
ZONE_IDENTIFIER_EXCLUDE_REGEX = r".*:Zone\.Identifier$"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sync benchmark results from GCS bucket to local directory."
    )
    p.add_argument(
        "--bucket-results-url",
        default=DEFAULT_BUCKET_RESULTS_URL,
        help=f"GCS source prefix (default: {DEFAULT_BUCKET_RESULTS_URL}).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_project_root() / "results",
        help="Local destination directory (default: ./results).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sync command without executing it.",
    )
    return p


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gcloud",
        "storage",
        "rsync",
        "-r",
        f"--exclude={ZONE_IDENTIFIER_EXCLUDE_REGEX}",
        args.bucket_results_url.rstrip("/"),
        str(output_dir),
    ]
    print("Running:", " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
