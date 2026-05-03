#!/usr/bin/env python3
"""Merge validated ``results/v1_replicates/<model>/chunk_*`` runs into ``results/v1``.

Each chunk is expected to contribute **one** completed run (first ``NNNN`` folder under the
chunk that has ``summary.csv``), typically ``0001``. Only runs passing validation are copied.

Destination folders ``results/v1/<model>/`` receive new sequential ``NNNN`` directories after
any existing numeric run dirs (``0001``, ``0002``, …). ``metadata.json`` is patched so
``run_index`` and ``run_subdir`` match the new folder name; ``run_group`` is preserved.

Examples::

    python scripts/analysis/integrate_v1_replicates.py --dry-run
    python scripts/analysis/integrate_v1_replicates.py
    python scripts/analysis/integrate_v1_replicates.py --models internvl35-4B qwen35-2B
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

FULL_BENCHMARK_TASKS = frozenset(
    {
        "egma-math",
        "matrix-reasoning",
        "mental-rotation",
        "theory-of-mind",
        "trog",
        "vocab",
    }
)


@dataclass
class ValidationIssue:
    path: str
    code: str
    detail: str = ""


@dataclass
class ModelReport:
    model_label: str
    chunks_ok: list[str] = field(default_factory=list)
    chunks_skipped: list[tuple[str, str]] = field(default_factory=list)
    destinations: list[tuple[str, str]] = field(default_factory=list)
    errors: list[ValidationIssue] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "results" / "v1_replicates",
        help="Folder containing per-model chunk trees (default: results/v1_replicates).",
    )
    p.add_argument(
        "--dest-root",
        type=Path,
        default=REPO_ROOT / "results" / "v1",
        help="LEVANTE v1 results root (default: results/v1).",
    )
    p.add_argument(
        "--models",
        nargs="*",
        help="Only integrate these model folder names (default: all non-old_* under source).",
    )
    p.add_argument(
        "--include-old-prefix",
        action="store_true",
        help="Include source folders whose names start with 'old_' (skipped by default).",
    )
    p.add_argument(
        "--cache-check",
        action="store_true",
        help=(
            "Verify cache/responses.json keys match trial_hash() for reconstructed trials "
            "(catches merged/stale caches)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print plan without copying or patching metadata.",
    )
    return p.parse_args()


def next_run_index(dest_model_dir: Path) -> int:
    if not dest_model_dir.exists():
        return 1
    best = 0
    for child in dest_model_dir.iterdir():
        if child.is_dir() and len(child.name) == 4 and child.name.isdigit():
            best = max(best, int(child.name))
    return best + 1


def validate_run_dir(run_dir: Path, *, cache_check: bool) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    rel = str(run_dir.relative_to(REPO_ROOT))

    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        issues.append(ValidationIssue(rel, "missing_metadata"))
        return issues

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(ValidationIssue(rel, "metadata_json_error", str(exc)))
        return issues

    if not (run_dir / "summary.csv").exists():
        issues.append(ValidationIssue(rel, "missing_summary_csv"))

    tasks = meta.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        issues.append(ValidationIssue(rel, "metadata_tasks_missing"))
    else:
        if frozenset(tasks) != FULL_BENCHMARK_TASKS:
            issues.append(
                ValidationIssue(
                    rel,
                    "unexpected_task_set",
                    f"have={sorted(tasks)}",
                )
            )
        for task_id in tasks:
            csv_path = run_dir / f"{task_id}.csv"
            if not csv_path.exists():
                issues.append(ValidationIssue(rel, "missing_task_csv", task_id))
            elif csv_path.stat().st_size < 32:
                issues.append(ValidationIssue(rel, "tiny_task_csv", task_id))

    if meta.get("dataset_version") != "v1":
        issues.append(
            ValidationIssue(rel, "unexpected_dataset_version", str(meta.get("dataset_version")))
        )

    if cache_check and issues == []:
        cache_issues = _validate_cache_keys(run_dir, meta)
        issues.extend(cache_issues)

    return issues


def _validate_cache_keys(run_dir: Path, meta: dict) -> list[ValidationIssue]:
    from levante_bench.config import get_task_def
    from levante_bench.evaluation.cache import trial_hash
    from levante_bench.tasks import get_task_dataset

    rel = str(run_dir.relative_to(REPO_ROOT))
    cache_path = run_dir / "cache" / "responses.json"
    if not cache_path.exists():
        return [ValidationIssue(rel, "missing_cache")]

    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return [ValidationIssue(rel, "cache_not_object")]

    version = meta["dataset_version"]
    tasks = list(meta["tasks"])
    true_random = bool(meta.get("true_random_option_order", False))
    run_seed = meta.get("run_seed")
    overrides = {
        "true_random_option_order": true_random,
        "prompt_language": str(meta.get("prompt_language", "en")),
    }
    if true_random and run_seed is not None:
        overrides["option_order_run_seed"] = run_seed

    trials: list[dict] = []
    for task_id in tasks:
        task_def = get_task_def(task_id, version, data_root=REPO_ROOT / "data", task_overrides=overrides)
        ds_cls = get_task_dataset(task_id)
        dataset = ds_cls(task_def=task_def, version=version, data_root=REPO_ROOT / "data")
        for i in range(len(dataset)):
            trial = dataset[i]
            trial["task_id"] = task_id
            trials.append(trial)

    need = {trial_hash(t) for t in trials}
    keys = set(raw)
    mi = len(need - keys)
    ex = len(keys - need)
    if mi or ex:
        return [
            ValidationIssue(
                rel,
                "cache_trial_hash_mismatch",
                f"missing_hashes={mi} extra_hashes={ex} need={len(need)} keys={len(keys)}",
            )
        ]
    return []


def pick_source_run(chunk_dir: Path) -> Path | None:
    runs = sorted([p for p in chunk_dir.glob("[0-9][0-9][0-9][0-9]") if p.is_dir()])
    for run_dir in runs:
        if (run_dir / "summary.csv").exists():
            return run_dir
    return None


def patch_metadata(dest_run: Path, new_index: int, *, dry_run: bool) -> None:
    meta_path = dest_run / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["run_index"] = new_index
    meta["run_subdir"] = f"{new_index:04d}"
    if dry_run:
        return
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def integrate_model(
    model_src: Path,
    model_dest: Path,
    *,
    cache_check: bool,
    dry_run: bool,
) -> ModelReport:
    label = model_src.name
    report = ModelReport(model_label=label)
    chunks = sorted(model_src.glob("chunk_*"))

    selected: list[tuple[Path, str]] = []
    for chunk in chunks:
        chosen = pick_source_run(chunk)
        if chosen is None:
            report.chunks_skipped.append((chunk.name, "no_completed_run_with_summary"))
            continue
        issues = validate_run_dir(chosen, cache_check=cache_check)
        if issues:
            report.errors.extend(issues)
            report.chunks_skipped.append((chunk.name, f"validation_failed:{chosen.name}"))
            continue
        selected.append((chosen, chunk.name))
        report.chunks_ok.append(chunk.name)

    if not selected:
        return report

    start = next_run_index(model_dest)
    if dry_run:
        idx = start
        for src, chunk_name in selected:
            report.destinations.append((str(src.relative_to(REPO_ROOT)), f"{idx:04d}"))
            idx += 1
        return report

    model_dest.mkdir(parents=True, exist_ok=True)
    idx = start
    for src, chunk_name in selected:
        dst = model_dest / f"{idx:04d}"
        if dst.exists():
            report.errors.append(
                ValidationIssue(str(dst.relative_to(REPO_ROOT)), "destination_exists")
            )
            break
        shutil.copytree(src, dst)
        patch_metadata(dst, idx, dry_run=False)
        report.destinations.append((str(src.relative_to(REPO_ROOT)), f"{idx:04d}"))
        idx += 1

    return report


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()

    if not source_root.is_dir():
        print(f"ERROR: source root not found: {source_root}", file=sys.stderr)
        return 2

    model_dirs = sorted([p for p in source_root.iterdir() if p.is_dir()])
    if not args.include_old_prefix:
        model_dirs = [p for p in model_dirs if not p.name.startswith("old_")]

    if args.models:
        wanted = set(args.models)
        model_dirs = [p for p in model_dirs if p.name in wanted]
        missing = wanted - {p.name for p in model_dirs}
        if missing:
            print(f"WARNING: unknown model folders (skipped): {sorted(missing)}", file=sys.stderr)

    reports: list[ModelReport] = []
    for model_src in model_dirs:
        model_dest = dest_root / model_src.name
        reports.append(
            integrate_model(
                model_src,
                model_dest,
                cache_check=args.cache_check,
                dry_run=args.dry_run,
            )
        )

    # ── stdout report ──────────────────────────────────────────────────────
    total_copied = sum(len(r.destinations) for r in reports)
    print(f"dry_run={args.dry_run} source_root={source_root} dest_root={dest_root}")
    print(f"models_scanned={len(model_dirs)} planned_integrations={total_copied}")
    print("")

    any_validation_errors = False
    for r in reports:
        print(f"## {r.model_label}")
        print(f"    chunks_ok={len(r.chunks_ok)} skipped_chunks={len(r.chunks_skipped)}")
        if r.chunks_ok:
            print(f"    integrated_chunks: {', '.join(r.chunks_ok)}")
        for ch, reason in r.chunks_skipped:
            print(f"    SKIP {ch}: {reason}")
        for src, dst in r.destinations:
            print(f"    -> {dst} from {src}")
        if r.errors:
            any_validation_errors = True
            print("    VALIDATION_ISSUES:")
            for issue in r.errors:
                extra = f" ({issue.detail})" if issue.detail else ""
                print(f"      - [{issue.code}] {issue.path}{extra}")
        print("")

    if any_validation_errors:
        print(
            "(validation issues listed above; those chunks were not integrated)",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
