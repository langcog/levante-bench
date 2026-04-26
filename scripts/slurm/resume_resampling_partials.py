#!/usr/bin/env python3
"""Resume partial true-random resampling runs in place.

This is intended for Marlowe preemptions. It scans existing run folders,
keeps completed task outputs, reuses each run's response cache, and evaluates
only tasks that do not already have a task CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tqdm import tqdm

from levante_bench.config import get_task_def, load_model_config, load_task_config
from levante_bench.config.defaults import detect_data_version
from levante_bench.data.loaders import load_human_proportions
from levante_bench.evaluation.adapters import postprocess_task_outputs
from levante_bench.evaluation.cache import load_cache, save_cache, trial_hash
from levante_bench.evaluation.human_comparison import annotate_human_metrics
from levante_bench.evaluation.outputs import (
    write_summary_csv,
    write_task_csv,
    write_task_npy,
)
from levante_bench.evaluation.runner import resolve_device
from levante_bench.runtime.modeling import build_model, resolve_model_config
from levante_bench.tasks import get_task_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume partial true-random run folders by evaluating missing tasks "
            "in place and reusing cache/responses.json."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help=(
            "Run folder, model root, or chunked model root to scan. Examples: "
            "results/resampling/qwen35-4B/v1/qwen35-4B or .../chunk_01/0004."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="Data root containing responses/assets (default: repo data/).",
    )
    parser.add_argument("--device", default="auto", help="Device: auto|cpu|cuda.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size. Defaults to each run metadata value.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override model max_new_tokens for resumed tasks.",
    )
    parser.add_argument(
        "--use-json-format",
        choices=["true", "false"],
        default=None,
        help="Override model use_json_format for resumed tasks.",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Only consider this task ID. Repeat for multiple tasks.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Resume at most this many partial runs.",
    )
    parser.add_argument(
        "--include-complete",
        action="store_true",
        help="Also inspect runs that already have summary.csv.",
    )
    parser.add_argument(
        "--overwrite-tasks",
        action="store_true",
        help="Recompute requested task CSV/NPY files even if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be resumed without loading models or writing outputs.",
    )
    return parser.parse_args()


def load_metadata(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "metadata.json").open(encoding="utf-8") as f:
        return json.load(f)


def find_run_dirs(run_root: Path, *, include_complete: bool) -> list[Path]:
    run_root = run_root.resolve()
    if (run_root / "metadata.json").exists():
        candidates = [run_root]
    else:
        candidates = sorted(p.parent for p in run_root.rglob("metadata.json"))

    run_dirs: list[Path] = []
    for run_dir in candidates:
        if not include_complete and (run_dir / "summary.csv").exists():
            continue
        metadata = load_metadata(run_dir)
        if not metadata.get("true_random_option_order"):
            continue
        if metadata.get("run_seed") is None:
            continue
        run_dirs.append(run_dir)
    return run_dirs


def parse_existing_task_accuracy(path: Path) -> float:
    total = 0
    correct = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            value = str(row.get("is_correct", "")).strip().lower()
            if value in {"true", "1", "yes"}:
                correct += 1
    return correct / total if total else 0.0


def task_is_complete(run_dir: Path, task_id: str) -> bool:
    return (run_dir / f"{task_id}.csv").exists() and (run_dir / f"{task_id}.npy").exists()


def run_tasks_from_metadata(metadata: dict[str, Any], requested: set[str] | None) -> list[str]:
    tasks = [str(task) for task in metadata.get("tasks") or []]
    if requested is not None:
        tasks = [task for task in tasks if task in requested]
    return tasks


def resolve_metadata_version(metadata: dict[str, Any], data_root: Path) -> str:
    raw_version = str(metadata.get("dataset_version") or "current")
    if raw_version.strip().lower() == "current":
        return detect_data_version(data_root)
    return raw_version


def evaluate_task(
    *,
    run_dir: Path,
    metadata: dict[str, Any],
    task_id: str,
    model: Any,
    model_name: str,
    model_cfg: dict[str, Any],
    version: str,
    data_root: Path,
    device: str,
    batch_size: int,
    prompt_language: str,
) -> float:
    task_cfg = load_task_config(task_id)
    if task_cfg is None:
        raise RuntimeError(f"No task config found for {task_id}")

    capabilities = model_cfg.get("capabilities", [])
    context_type = task_cfg.get("context_type", "none")
    if capabilities and context_type not in capabilities and context_type != "none":
        raise RuntimeError(
            f"Model {model_name} lacks capability {context_type!r} for task {task_id}"
        )

    run_seed = int(metadata["run_seed"])
    overrides = {
        "true_random_option_order": True,
        "option_order_run_seed": run_seed,
        "prompt_language": prompt_language,
    }
    task_def = get_task_def(
        task_id,
        version,
        data_root=data_root,
        task_overrides=overrides,
    )
    if task_def is None:
        raise RuntimeError(f"No task def for task={task_id} version={version}")

    dataset_cls = get_task_dataset(task_id)
    if dataset_cls is None:
        raise RuntimeError(f"No dataset registered for {task_id}")

    dataset = dataset_cls(task_def=task_def, version=version, data_root=data_root)
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset for {task_id}")

    human_props: dict = {}
    if task_def.human_response_path and Path(task_def.human_response_path).exists():
        human_props = load_human_proportions(task_def.human_response_path)

    cache_path = run_dir / "cache" / "responses.json"
    cache = load_cache(cache_path)
    task_results: list[dict[str, Any]] = []
    task_trials: list[dict[str, Any]] = []
    max_new_tokens = model_cfg.get("max_new_tokens", 64)

    for chunk_start in tqdm(
        range(0, len(dataset), batch_size),
        desc=f"  {run_dir.name}/{task_id}",
        unit="batch",
    ):
        chunk_trials: list[dict[str, Any]] = []
        chunk_hashes: list[str] = []
        chunk_results: list[dict[str, Any] | None] = []
        uncached_positions: list[int] = []
        uncached_trials: list[dict[str, Any]] = []

        chunk_end = min(chunk_start + batch_size, len(dataset))
        for i in range(chunk_start, chunk_end):
            trial = dataset[i]
            task_trials.append(trial)
            trial["task_id"] = task_id
            trial["max_new_tokens"] = max_new_tokens
            h = trial_hash(trial)
            chunk_trials.append(trial)
            chunk_hashes.append(h)

            cached = cache.get(h)
            if cached is not None:
                if "option_order_seed" not in cached and trial.get("option_order_seed") is not None:
                    cached["option_order_seed"] = str(trial.get("option_order_seed"))
                    cache[h] = cached
                    save_cache(cache_path, cache)
                chunk_results.append(cached)
            else:
                chunk_results.append(None)
                uncached_positions.append(len(chunk_results) - 1)
                uncached_trials.append(trial)

        if uncached_trials:
            uncached_results = model.evaluate_trials_batch(uncached_trials)
            if len(uncached_results) != len(uncached_trials):
                raise RuntimeError(
                    f"Model {model_name} returned {len(uncached_results)} results "
                    f"for {len(uncached_trials)} trials."
                )
            for pos, result in zip(uncached_positions, uncached_results):
                trial_for_result = chunk_trials[pos]
                result["option_order_seed"] = str(
                    trial_for_result.get("option_order_seed", "")
                )
                if human_props:
                    annotate_human_metrics(result, human_props.get(result["item_uid"]))
                h = chunk_hashes[pos]
                cache[h] = result
                save_cache(cache_path, cache)
                chunk_results[pos] = result

        task_results.extend([r for r in chunk_results if r is not None])

    write_task_csv(run_dir, task_id, task_results)
    write_task_npy(run_dir, task_id, task_results, task_trials=task_trials)
    for out_path in postprocess_task_outputs(
        task_id=task_id,
        model_dir=run_dir,
        task_results=task_results,
        task_trials=task_trials,
    ):
        print(f"  {task_id}: wrote {out_path}")

    correct = sum(1 for result in task_results if result["is_correct"])
    accuracy = correct / len(task_results) if task_results else 0.0
    print(f"  {run_dir}/{task_id}: {accuracy:.4f} ({correct}/{len(task_results)})")
    return accuracy


def model_key(metadata: dict[str, Any]) -> tuple[str, str | None]:
    model_name = str(metadata.get("model") or "").strip()
    model_size = metadata.get("model_size")
    model_size = str(model_size).strip() if model_size is not None else None
    return model_name, model_size or None


def build_model_cfg(
    *,
    model_name: str,
    model_size: str | None,
    max_new_tokens: int | None,
    use_json_format: str | None,
) -> dict[str, Any]:
    base_cfg = load_model_config(model_name)
    if base_cfg is None:
        raise RuntimeError(f"No model config found for {model_name}")
    overrides: dict[str, Any] = {}
    if model_size:
        overrides["size"] = model_size
    if max_new_tokens is not None:
        overrides["max_new_tokens"] = max_new_tokens
    if use_json_format is not None:
        overrides["use_json_format"] = use_json_format == "true"
    return resolve_model_config(
        model_name=model_name,
        model_overrides=overrides,
        model_config=base_cfg,
    )


def summarize_existing_tasks(run_dir: Path, tasks: list[str]) -> dict[str, float]:
    accuracies: dict[str, float] = {}
    for task_id in tasks:
        path = run_dir / f"{task_id}.csv"
        if path.exists():
            accuracies[task_id] = parse_existing_task_accuracy(path)
    return accuracies


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    data_root = args.data_root.resolve()
    requested_tasks = set(args.tasks) if args.tasks else None
    device = resolve_device(args.device)

    run_dirs = find_run_dirs(run_root, include_complete=args.include_complete)
    if args.max_runs is not None:
        run_dirs = run_dirs[: max(0, args.max_runs)]

    if not run_dirs:
        print(f"No resumable partial true-random runs found under {run_root}")
        return 0

    plans: list[tuple[Path, dict[str, Any], list[str], list[str]]] = []
    for run_dir in run_dirs:
        metadata = load_metadata(run_dir)
        tasks = run_tasks_from_metadata(metadata, requested_tasks)
        missing = [
            task for task in tasks if args.overwrite_tasks or not task_is_complete(run_dir, task)
        ]
        if missing:
            plans.append((run_dir, metadata, tasks, missing))

    if not plans:
        print("No missing task outputs found in selected runs.")
        return 0

    print(f"run_root={run_root}")
    print(f"resumable_runs={len(plans)}")
    for run_dir, metadata, _, missing in plans:
        print(
            f"  {run_dir}: model={metadata.get('model')}-{metadata.get('model_size')} "
            f"seed={metadata.get('run_seed')} missing={','.join(missing)}"
        )

    if args.dry_run:
        return 0

    grouped: dict[tuple[str, str | None], list[tuple[Path, dict[str, Any], list[str], list[str]]]] = defaultdict(list)
    for plan in plans:
        grouped[model_key(plan[1])].append(plan)

    for (model_name, model_size), group in grouped.items():
        if not model_name:
            raise RuntimeError("Run metadata is missing model name")
        model_cfg = build_model_cfg(
            model_name=model_name,
            model_size=model_size,
            max_new_tokens=args.max_new_tokens,
            use_json_format=args.use_json_format,
        )
        print(f"Loading model {model_name}-{model_size or ''} on {device}...")
        model = build_model(
            model_name=model_name,
            model_cfg=model_cfg,
            device=device,
            auto_load=True,
        )

        for run_dir, metadata, tasks, missing in group:
            version = resolve_metadata_version(metadata, data_root)
            batch_size = max(1, int(args.batch_size or metadata.get("batch_size") or 1))
            prompt_language = str(metadata.get("prompt_language") or "en")
            task_accuracies = summarize_existing_tasks(run_dir, tasks)

            for task_id in missing:
                task_accuracies[task_id] = evaluate_task(
                    run_dir=run_dir,
                    metadata=metadata,
                    task_id=task_id,
                    model=model,
                    model_name=model_name,
                    model_cfg=model_cfg,
                    version=version,
                    data_root=data_root,
                    device=device,
                    batch_size=batch_size,
                    prompt_language=prompt_language,
                )

            if all((run_dir / f"{task_id}.csv").exists() for task_id in tasks):
                ordered = {task: task_accuracies[task] for task in tasks if task in task_accuracies}
                summary_path = write_summary_csv(run_dir, ordered)
                print(f"  Summary: {summary_path}")
            else:
                still_missing = [
                    task for task in tasks if not (run_dir / f"{task}.csv").exists()
                ]
                print(f"  Still incomplete: {run_dir} missing={','.join(still_missing)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
