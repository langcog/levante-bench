#!/usr/bin/env python3
"""Download synthetic vocab assets from GCS into data/assets/synth_vocab."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_BUCKET_URI = "gs://levante-bench/corpus_data/synth_vocab"


def run(
    *,
    bucket_uri: str = DEFAULT_BUCKET_URI,
    output_dir: Path | None = None,
    dry_run: bool = False,
    delete_unmatched: bool = False,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = output_dir or repo_root / "data" / "assets" / "synth_vocab"
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gcloud",
        "storage",
        "rsync",
        "--recursive",
        bucket_uri.rstrip("/"),
        str(output_dir),
    ]
    if delete_unmatched:
        cmd.insert(3, "--delete-unmatched-destination-objects")

    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)
    print(f"Downloaded synthetic vocab assets to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download synthetic vocab assets from the LEVANTE bucket."
    )
    parser.add_argument(
        "--bucket-uri",
        default=DEFAULT_BUCKET_URI,
        help=f"GCS prefix to sync from (default: {DEFAULT_BUCKET_URI}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local output directory (default: data/assets/synth_vocab).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the rsync command only.")
    parser.add_argument(
        "--delete-unmatched",
        action="store_true",
        help="Delete local files under output-dir that are not present in the bucket.",
    )
    args = parser.parse_args()

    run(
        bucket_uri=args.bucket_uri,
        output_dir=args.output_dir,
        dry_run=bool(args.dry_run),
        delete_unmatched=bool(args.delete_unmatched),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
