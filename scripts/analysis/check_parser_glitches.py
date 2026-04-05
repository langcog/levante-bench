#!/usr/bin/env python3
"""Scan result CSVs for parser glitches and suggest targeted fixes.

This script inspects trial-level result CSV files under results/ and reports:
- high unparseable / parse_confidence=none rates
- suspicious raw output patterns (e.g., punctuation-wrapped labels)
- potential parser/prompt fixes by model+task cluster
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PUNCTUATED_LABEL_RE = re.compile(r"^[\s\W_]*([A-Za-z])[\s\W_]*$")
ANSWER_PHRASE_RE = re.compile(r"\banswer\s*[:=]\s*([A-Za-z])\b", re.IGNORECASE)
OPTION_PHRASE_RE = re.compile(r"\b(?:option|answer)\s+is\s+([A-Za-z])\b", re.IGNORECASE)


@dataclass
class FileStats:
    path: Path
    version: str
    model_tag: str
    task: str
    total_rows: int
    unparseable_rows: int
    parse_none_rows: int
    empty_prediction_rows: int
    parse_method_counts: Counter[str]
    confidence_counts: Counter[str]
    pattern_counts: Counter[str]
    sample_rows: dict[str, list[dict[str, str]]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect trial CSVs for parser glitches and write diagnostics with fix suggestions."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Results root directory (default: results).",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--local",
        dest="source_mode",
        action="store_const",
        const="local",
        help="Analyze local results-root (default source is bucket).",
    )
    source_group.add_argument(
        "--bucket",
        dest="source_mode",
        action="store_const",
        const="bucket",
        help="Stage results from bucket before analysis (default).",
    )
    parser.set_defaults(source_mode="bucket")
    parser.add_argument(
        "--bucket-results-url",
        default="gs://levante-bench/results",
        help="Bucket prefix to stage when using --bucket (default: gs://levante-bench/results).",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path(".tmp/parser-glitch-audit-staging"),
        help="Local staging directory for bucket mode (default: .tmp/parser-glitch-audit-staging).",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep staged bucket snapshot after the run.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/parser-glitch-report.json"),
        help="Output JSON report path (default: results/parser-glitch-report.json).",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("results/parser-glitch-report.md"),
        help="Output Markdown report path (default: results/parser-glitch-report.md).",
    )
    parser.add_argument(
        "--warn-unparseable-rate",
        type=float,
        default=0.10,
        help="Warn when unparseable rate exceeds this threshold (default: 0.10).",
    )
    parser.add_argument(
        "--warn-empty-pred-rate",
        type=float,
        default=0.10,
        help="Warn when empty prediction rate exceeds this threshold (default: 0.10).",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=10,
        help="Minimum rows required before warning (default: 10).",
    )
    parser.add_argument(
        "--max-examples-per-issue",
        type=int,
        default=5,
        help="Max example rows to include per issue bucket (default: 5).",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("data/assets/manifest.csv"),
        help="Manifest CSV used to resolve prompts by trial_id/item_uid (default: data/assets/manifest.csv).",
    )
    return parser.parse_args()


def _is_trial_csv(path: Path) -> bool:
    name = path.name
    if name in {"summary.csv", "model-comparison-report.csv"}:
        return False
    if name.endswith("-by-type.csv"):
        return False
    return True


def _infer_version_model_task(path: Path, results_root: Path) -> tuple[str, str, str]:
    rel = path.relative_to(results_root)
    parts = rel.parts
    if len(parts) >= 3:
        version = parts[0]
        model_tag = parts[1]
    else:
        version = "unknown"
        model_tag = "unknown"
    task = path.stem
    return version, model_tag, task


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _row_preview(
    row: dict[str, str],
    prompt_lookup: dict[str, str],
) -> dict[str, str]:
    predicted_label = (row.get("predicted_label") or "").strip()
    predicted_value = (row.get("predicted_value") or "").strip()
    if predicted_label and predicted_value:
        parsed_prediction = f"label={predicted_label}, value={predicted_value}"
    elif predicted_label:
        parsed_prediction = predicted_label
    elif predicted_value:
        parsed_prediction = predicted_value
    else:
        parsed_prediction = ""
    trial_id = (row.get("trial_id") or "").strip()
    item_uid = (row.get("item_uid") or "").strip()
    prompt = prompt_lookup.get(trial_id) or prompt_lookup.get(item_uid) or ""
    return {
        "trial_id": trial_id,
        "item_uid": item_uid,
        "prompt": prompt,
        "parsed_prediction": parsed_prediction,
        "prediction": row.get("generated_text", ""),
        "parse_method": row.get("parse_method", ""),
        "parse_confidence": row.get("parse_confidence", ""),
        "correct_label": row.get("correct_label", ""),
    }


def _analyze_file(
    path: Path,
    results_root: Path,
    max_examples: int,
    prompt_lookup: dict[str, str],
) -> FileStats | None:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    if "generated_text" not in fields:
        return None

    version, model_tag, task = _infer_version_model_task(path, results_root)
    method_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)

    unparseable_rows = 0
    parse_none_rows = 0
    empty_prediction_rows = 0

    def add_sample(issue: str, row: dict[str, str]) -> None:
        if len(samples[issue]) < max_examples:
            samples[issue].append(_row_preview(row, prompt_lookup))

    for row in rows:
        parse_method = (row.get("parse_method") or "").strip() or "missing"
        parse_conf = (row.get("parse_confidence") or "").strip() or "missing"
        pred_label = (row.get("predicted_label") or "").strip()
        pred_value = (row.get("predicted_value") or "").strip()
        text = (row.get("generated_text") or "").strip()
        method_counts[parse_method] += 1
        confidence_counts[parse_conf] += 1

        if parse_method == "unparseable":
            unparseable_rows += 1
        if parse_conf == "none":
            parse_none_rows += 1
        if not pred_label and not pred_value:
            empty_prediction_rows += 1

        if parse_method == "unparseable":
            if PUNCTUATED_LABEL_RE.match(text or ""):
                pattern_counts["punctuated_single_label"] += 1
                add_sample("punctuated_single_label", row)
            if ANSWER_PHRASE_RE.search(text) or OPTION_PHRASE_RE.search(text):
                pattern_counts["answer_phrase_unparsed"] += 1
                add_sample("answer_phrase_unparsed", row)
            if len(text) > 120:
                pattern_counts["long_unparseable_text"] += 1
                add_sample("long_unparseable_text", row)
            if text:
                pattern_counts["nonempty_unparseable"] += 1
                add_sample("nonempty_unparseable", row)
            else:
                pattern_counts["empty_generated_text"] += 1
                add_sample("empty_generated_text", row)

    return FileStats(
        path=path,
        version=version,
        model_tag=model_tag,
        task=task,
        total_rows=len(rows),
        unparseable_rows=unparseable_rows,
        parse_none_rows=parse_none_rows,
        empty_prediction_rows=empty_prediction_rows,
        parse_method_counts=method_counts,
        confidence_counts=confidence_counts,
        pattern_counts=pattern_counts,
        sample_rows=dict(samples),
    )


def _issue_suggestions(aggregate: dict[str, Any]) -> list[str]:
    suggestions: list[str] = []
    patterns = aggregate["pattern_counts"]
    if patterns.get("punctuated_single_label", 0) > 0:
        suggestions.append(
            "Parser rule: accept punctuation-wrapped single labels (e.g., '; A:')."
        )
    if patterns.get("answer_phrase_unparsed", 0) > 0:
        suggestions.append(
            "Parser rule: broaden explicit answer phrase matching for label extraction."
        )
    if patterns.get("long_unparseable_text", 0) > 0:
        suggestions.append(
            "Prompt tweak: append strict short-answer suffix (single label only)."
        )
    if aggregate["unparseable_rate"] >= 0.25:
        suggestions.append(
            "Model config: enforce structured output (JSON answer field) where supported."
        )
    if aggregate["empty_prediction_rate"] >= 0.20:
        suggestions.append(
            "Generation settings: review max_new_tokens / stop tokens for truncation artifacts."
        )
    if not suggestions:
        suggestions.append("No obvious parser glitch pattern detected.")
    return suggestions


def _build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Parser Glitch Audit")
    lines.append("")
    lines.append(f"- Generated: `{report['generated_at']}`")
    lines.append(f"- Source mode: `{report['source_mode']}`")
    lines.append(f"- Source URI: `{report['source_uri']}`")
    lines.append(f"- Scan root: `{report['scan_root']}`")
    lines.append(f"- Trial CSV files scanned: `{report['totals']['files_scanned']}`")
    lines.append(f"- Trial rows scanned: `{report['totals']['rows_scanned']}`")
    lines.append("")
    lines.append("## Top Risk Clusters")
    lines.append("")
    if not report["clusters"]:
        lines.append("No parser-risk clusters found.")
        return "\n".join(lines) + "\n"

    for cluster in report["clusters"]:
        lines.append(
            f"### `{cluster['version']}/{cluster['model_tag']}/{cluster['task']}`"
        )
        lines.append(
            f"- Rows: `{cluster['rows']}` | Unparseable: `{cluster['unparseable_rows']}` "
            f"(`{cluster['unparseable_rate']:.1%}`) | Empty predictions: "
            f"`{cluster['empty_prediction_rows']}` (`{cluster['empty_prediction_rate']:.1%}`)"
        )
        lines.append("- Suggested fixes:")
        for suggestion in cluster["suggestions"]:
            lines.append(f"  - {suggestion}")

        examples = cluster.get("samples", {})
        if examples:
            lines.append("- Example rows:")
            for issue, rows in examples.items():
                lines.append(f"  - `{issue}`:")
                for row in rows[:3]:
                    lines.append(
                        "    - "
                        f"trial=`{row.get('trial_id', '')}` "
                        f"parsed_prediction=`{row.get('parsed_prediction', '')}` "
                        f"parse=`{row.get('parse_method', '')}`"
                    )
                    lines.append("      Prompt:")
                    lines.append("      ```text")
                    lines.append(row.get("prompt", ""))
                    lines.append("      ```")
                    lines.append("      Prediction:")
                    lines.append("      ```text")
                    lines.append(row.get("prediction", ""))
                    lines.append("      ```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load_prompt_lookup(manifest_csv: Path) -> dict[str, str]:
    if not manifest_csv.exists():
        return {}
    lookup: dict[str, str] = {}
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key_trial = (row.get("item_uid") or "").strip()
            if not key_trial:
                continue
            prompt = (row.get("prompt") or "").strip()
            prompt_phrase = (row.get("prompt_phrase") or "").strip()
            full_prompt = (row.get("full_prompt") or "").strip()
            parts = [p for p in (prompt, prompt_phrase, full_prompt) if p and p.upper() != "NA"]
            rendered = "\n".join(parts)
            if rendered:
                lookup[key_trial] = rendered
    return lookup


def main() -> int:
    args = _parse_args()
    source_mode = str(args.source_mode)
    manifest_csv = args.manifest_csv.resolve()
    prompt_lookup = _load_prompt_lookup(manifest_csv)

    if source_mode == "local":
        scan_root = args.results_root.resolve()
        source_uri = str(scan_root)
    else:
        staging_dir = args.staging_dir.resolve()
        staged_results_root = staging_dir / "results"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staged_results_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            "gcloud",
            "storage",
            "rsync",
            "-r",
            args.bucket_results_url.rstrip("/"),
            str(staged_results_root),
        ]
        print("Running:", " ".join(cmd))
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            return completed.returncode
        scan_root = staged_results_root
        source_uri = args.bucket_results_url.rstrip("/")

    files = sorted(p for p in scan_root.rglob("*.csv") if p.is_file() and _is_trial_csv(p))

    all_stats: list[FileStats] = []
    for path in files:
        stats = _analyze_file(path, scan_root, args.max_examples_per_issue, prompt_lookup)
        if stats is not None:
            all_stats.append(stats)

    clusters = []
    total_rows = 0
    total_unparseable = 0
    total_empty = 0

    for stats in all_stats:
        total_rows += stats.total_rows
        total_unparseable += stats.unparseable_rows
        total_empty += stats.empty_prediction_rows

        unparseable_rate = _safe_rate(stats.unparseable_rows, stats.total_rows)
        empty_rate = _safe_rate(stats.empty_prediction_rows, stats.total_rows)
        if (
            stats.total_rows < args.min_rows
            or (
                unparseable_rate < args.warn_unparseable_rate
                and empty_rate < args.warn_empty_pred_rate
            )
        ):
            continue

        cluster = {
            "path": str(stats.path),
            "version": stats.version,
            "model_tag": stats.model_tag,
            "task": stats.task,
            "rows": stats.total_rows,
            "unparseable_rows": stats.unparseable_rows,
            "unparseable_rate": unparseable_rate,
            "parse_none_rows": stats.parse_none_rows,
            "parse_none_rate": _safe_rate(stats.parse_none_rows, stats.total_rows),
            "empty_prediction_rows": stats.empty_prediction_rows,
            "empty_prediction_rate": empty_rate,
            "parse_method_counts": dict(stats.parse_method_counts),
            "confidence_counts": dict(stats.confidence_counts),
            "pattern_counts": dict(stats.pattern_counts),
            "samples": stats.sample_rows,
        }
        cluster["suggestions"] = _issue_suggestions(cluster)
        clusters.append(cluster)

    clusters.sort(
        key=lambda c: (c["unparseable_rate"], c["empty_prediction_rate"], c["rows"]),
        reverse=True,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_mode": source_mode,
        "source_uri": source_uri,
        "scan_root": str(scan_root),
        "manifest_csv": str(manifest_csv),
        "prompt_lookup_size": len(prompt_lookup),
        "thresholds": {
            "warn_unparseable_rate": args.warn_unparseable_rate,
            "warn_empty_pred_rate": args.warn_empty_pred_rate,
            "min_rows": args.min_rows,
        },
        "totals": {
            "files_scanned": len(all_stats),
            "rows_scanned": total_rows,
            "unparseable_rows": total_unparseable,
            "unparseable_rate": _safe_rate(total_unparseable, total_rows),
            "empty_prediction_rows": total_empty,
            "empty_prediction_rate": _safe_rate(total_empty, total_rows),
        },
        "clusters": clusters,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_build_markdown(report), encoding="utf-8")

    print(f"Wrote JSON report: {args.output_json}")
    print(f"Wrote Markdown report: {args.output_markdown}")
    print(
        f"Scanned {report['totals']['files_scanned']} files, "
        f"{report['totals']['rows_scanned']} rows."
    )
    print(f"Risk clusters: {len(clusters)}")
    if source_mode == "bucket" and not args.keep_staging:
        shutil.rmtree(args.staging_dir.resolve(), ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
