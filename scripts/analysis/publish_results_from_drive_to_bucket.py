#!/usr/bin/env python3
"""Publish benchmark results from Drive to GCS for dashboard consumption.

Pipeline:
1) Download results folder from Google Drive into a local staging directory.
2) Build model-comparison-report.json from staged results.
3) Sync staged results to gs://<bucket>/results.
4) Mark report JSON as no-cache so dashboard refreshes pull latest data.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_FOLDER_URL = "https://drive.google.com/drive/folders/1NwlA8huu9GoXuwUnq0TQI6MwPCaX779C?usp=sharing"
DEFAULT_BUCKET_RESULTS_URL = "gs://levante-bench/results"
ZONE_IDENTIFIER_EXCLUDE_REGEX = r".*:Zone\.Identifier$"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _run(cmd: list[str], dry_run: bool = False) -> int:
    print("Running:", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def _staged_summary_count(staged_results_root: Path) -> int:
    return sum(1 for _ in staged_results_root.rglob("summary.csv"))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Download benchmark results from Drive, build report JSON, and "
            "publish to a GCS bucket prefix."
        )
    )
    p.add_argument(
        "--folder-url",
        default=DEFAULT_FOLDER_URL,
        help=f"Google Drive folder URL (default: {DEFAULT_FOLDER_URL}).",
    )
    p.add_argument(
        "--bucket-results-url",
        default=DEFAULT_BUCKET_RESULTS_URL,
        help=f"GCS destination prefix (default: {DEFAULT_BUCKET_RESULTS_URL}).",
    )
    p.add_argument(
        "--staging-dir",
        type=Path,
        default=_project_root() / ".tmp" / "results-publish-staging",
        help="Local staging directory (default: .tmp/results-publish-staging).",
    )
    p.add_argument(
        "--remaining-ok",
        action="store_true",
        help="Pass --remaining-ok to gdown folder download.",
    )
    p.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep staging directory after publish.",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Drive download and use existing staging contents.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    p.add_argument(
        "--min-summary-files",
        type=int,
        default=1,
        help=(
            "Minimum number of summary.csv files required in staging before publish "
            "(default: 1)."
        ),
    )
    p.add_argument(
        "--fallback-results-root",
        type=Path,
        default=_project_root() / "results",
        help=(
            "Local results root to use as fallback source when Drive download fails "
            "under --remaining-ok (default: ./results)."
        ),
    )
    return p


def main() -> int:
    args = _parser().parse_args()
    root = _project_root()

    staging_dir = args.staging_dir.resolve()
    staged_results_root = staging_dir / "results"
    staged_report_json = staged_results_root / "model-comparison-report.json"
    bucket_results_url = args.bucket_results_url.rstrip("/")
    report_url = f"{bucket_results_url}/model-comparison-report.json"

    if not args.skip_download and staging_dir.exists():
        if args.dry_run:
            print(f"Would remove existing staging directory: {staging_dir}")
        else:
            shutil.rmtree(staging_dir)

    staged_results_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        cmd = [
            sys.executable,
            str(root / "scripts" / "analysis" / "download_results_from_drive.py"),
            "--folder-url",
            args.folder_url,
            "--output-dir",
            str(staged_results_root),
        ]
        if args.remaining_ok:
            cmd.append("--remaining-ok")
        rc = _run(cmd, dry_run=args.dry_run)
        if rc != 0:
            summary_count = _staged_summary_count(staged_results_root)
            if args.remaining_ok and summary_count >= args.min_summary_files:
                print(
                    "Drive download reported errors, but --remaining-ok is set and "
                    f"{summary_count} summary file(s) were staged. Continuing publish.",
                    file=sys.stderr,
                )
            else:
                if args.remaining_ok:
                    fallback_root = args.fallback_results_root.resolve()
                    fallback_summary_count = _staged_summary_count(fallback_root)
                    if fallback_summary_count >= args.min_summary_files:
                        print(
                            "Drive download failed; using fallback local results root "
                            f"{fallback_root} with {fallback_summary_count} summary file(s).",
                            file=sys.stderr,
                        )
                        if args.dry_run:
                            print(
                                f"Would copy fallback results from {fallback_root} into {staged_results_root}"
                            )
                        else:
                            staged_results_root.mkdir(parents=True, exist_ok=True)
                            for entry in fallback_root.iterdir():
                                dest = staged_results_root / entry.name
                                if entry.is_dir():
                                    shutil.copytree(entry, dest, dirs_exist_ok=True)
                                else:
                                    shutil.copy2(entry, dest)
                    else:
                        print(
                            "Drive download failed and fallback local results do not meet "
                            f"--min-summary-files={args.min_summary_files}.",
                            file=sys.stderr,
                        )
                        return rc
                else:
                    print(
                        "Drive download failed. Re-run with --remaining-ok or verify Drive permissions.",
                        file=sys.stderr,
                    )
                    return rc

    summary_count = _staged_summary_count(staged_results_root)
    if summary_count < args.min_summary_files:
        print(
            f"Staging contains only {summary_count} summary file(s), below --min-summary-files={args.min_summary_files}.",
            file=sys.stderr,
        )
        return 2

    rc = _run(
        [
            sys.executable,
            str(root / "scripts" / "analysis" / "build_model_comparison_report.py"),
            "--results-root",
            str(staged_results_root),
            "--output-json",
            str(staged_report_json),
        ],
        dry_run=args.dry_run,
    )
    if rc != 0:
        return rc

    rc = _run(
        [
            "gcloud",
            "storage",
            "rsync",
            "-r",
            f"--exclude={ZONE_IDENTIFIER_EXCLUDE_REGEX}",
            str(staged_results_root),
            bucket_results_url,
        ],
        dry_run=args.dry_run,
    )
    if rc != 0:
        return rc

    # Reduce stale dashboard reads: force report JSON to no-cache.
    rc = _run(
        [
            "gcloud",
            "storage",
            "objects",
            "update",
            report_url,
            "--cache-control=no-store,max-age=0",
        ],
        dry_run=args.dry_run,
    )
    if rc != 0:
        return rc

    print("")
    print("Publish complete.")
    print(f"Bucket prefix: {bucket_results_url}")
    print(f"Report URL: {report_url}")
    print("Set Vercel RESULTS_REPORT_URL to this report URL (if not already set).")

    if not args.keep_staging:
        if args.dry_run:
            print(f"Would remove staging directory: {staging_dir}")
        else:
            shutil.rmtree(staging_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

