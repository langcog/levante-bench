#!/usr/bin/env python3
"""Stitch chunked true-random runs into a single 0001..NNNN sequence.

Expected source layout (from submit_resampling_100.sh):
  <source-root>/chunk_01/0001
  <source-root>/chunk_01/0002
  ...
  <source-root>/chunk_10/0010

Output layout:
  <output-root>/0001
  <output-root>/0002
  ...
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stitch chunked run folders into sequential runs.")
    p.add_argument("--source-root", required=True, help="Root containing chunk_* folders.")
    p.add_argument("--output-root", required=True, help="Destination root for 0001..NNNN folders.")
    p.add_argument(
        "--copy-mode",
        default="copy",
        choices=["copy", "symlink"],
        help="How to materialize stitched runs (default: copy).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    chunks = sorted([p for p in source_root.glob("chunk_*") if p.is_dir()])
    if not chunks:
        raise RuntimeError(f"No chunk_* directories found under: {source_root}")

    run_dirs: list[Path] = []
    for chunk in chunks:
        runs = sorted([p for p in chunk.glob("[0-9][0-9][0-9][0-9]") if p.is_dir()])
        run_dirs.extend(runs)

    if not run_dirs:
        raise RuntimeError(f"No run directories found under chunks in: {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(run_dirs, start=1):
        dst = output_root / f"{i:04d}"
        if dst.exists():
            raise FileExistsError(f"Destination already exists: {dst}")
        if args.copy_mode == "symlink":
            dst.symlink_to(src, target_is_directory=True)
        else:
            shutil.copytree(src, dst)

    print(f"stitched_runs={len(run_dirs)}")
    print(f"source_root={source_root}")
    print(f"output_root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
