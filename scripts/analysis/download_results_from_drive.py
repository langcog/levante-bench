#!/usr/bin/env python3
"""
Download benchmark result artifacts from a public Google Drive folder.

This script is a thin wrapper around `gdown` folder download support so we can
keep a consistent, documented command in this repo.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

DEFAULT_RESULTS_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1aaWa4kDqdl9uNQF_ijXe7SSRYh54UJ-N"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download all shared benchmark results from Google Drive into results/."
    )
    parser.add_argument(
        "--folder-url",
        default=DEFAULT_RESULTS_FOLDER_URL,
        help="Google Drive folder URL (default: LEVANTE shared results folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_project_root() / "results",
        help="Local destination directory (default: ./results).",
    )
    parser.add_argument(
        "--remaining-ok",
        action="store_true",
        help="Continue even if some files in the folder are inaccessible.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed without running it.",
    )
    return parser


def _ensure_gdown_installed() -> None:
    if importlib.util.find_spec("gdown") is None:
        raise RuntimeError(
            "gdown is required but not installed.\n"
            "Install it with:\n"
            "  .venv/bin/python -m pip install gdown"
        )


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        args.folder_url,
        "-O",
        str(output_dir),
    ]
    if args.remaining_ok:
        cmd.append("--remaining-ok")

    print("Running:", " ".join(cmd))
    if args.dry_run:
        return 0

    _ensure_gdown_installed()
    completed = subprocess.run(cmd)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

