"""Run evaluation: for each model, evaluate all tasks, write results."""

import json
import os
import random
import sys
from pathlib import Path

from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf

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
from levante_bench.runtime.modeling import build_model, resolve_model_config
from levante_bench.tasks import get_task_dataset


def _two_letter_language_code(language: object) -> str | None:
    """Extract a normalized 2-letter language code from a locale string."""
    value = str(language or "").strip()
    if not value:
        return None
    primary = value.split("-", 1)[0].split("_", 1)[0].lower()
    letters = "".join(ch for ch in primary if ch.isalpha())
    if len(letters) < 2:
        return None
    return letters[:2]


def _results_language_suffix(task_overrides: dict) -> str:
    """Return folder suffix (e.g., '-de') for non-English prompt language."""
    global_overrides = task_overrides.get("__all__", {})
    if not isinstance(global_overrides, dict):
        return ""
    code = _two_letter_language_code(global_overrides.get("prompt_language"))
    if not code or code == "en":
        return ""
    return f"-{code}"


def _prompt_language(task_overrides: dict) -> str:
    global_overrides = task_overrides.get("__all__", {})
    if not isinstance(global_overrides, dict):
        return "en"
    return str(global_overrides.get("prompt_language") or "en")


def _normalize_output_base(output_base: Path, version: str) -> Path:
    """Normalize legacy output_dir values to canonical results root.

    Handles legacy config values like:
      - results/<version>
      - results/<model>-<version>
    so runner always writes to results/<version>/<model-size[-lang]>.
    """
    normalized = output_base

    # If user passed results/<version>, collapse to results.
    if normalized.name == version:
        normalized = normalized.parent

    # Legacy experiment configs used results/<model>-<version>.
    if (
        normalized.parent.name == "results"
        and normalized.name.lower().endswith(f"-{version.lower()}")
    ):
        normalized = normalized.parent

    return normalized


def resolve_device(device: str) -> str:
    """Resolve auto device selection with safe CUDA -> CPU fallback."""
    choice = (device or "auto").strip().lower()
    if choice != "auto":
        return choice
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _slurm_run_label() -> str | None:
    """Build a unique run label from Slurm environment variables."""
    job_id = str(os.getenv("SLURM_JOB_ID") or "").strip()
    if not job_id:
        return None
    task_id = str(os.getenv("SLURM_ARRAY_TASK_ID") or "").strip()
    proc_id = str(os.getenv("SLURM_PROCID") or "").strip()
    if task_id:
        return f"job{job_id}-task{task_id}"
    if proc_id:
        return f"job{job_id}-proc{proc_id}"
    return f"job{job_id}"


def _resolve_run_label(
    *,
    cfg: DictConfig,
    true_random_option_order: bool,
) -> str | None:
    """Resolve optional run-group label (parent folder) for true-random runs."""
    if not true_random_option_order:
        return None

    run_label_cfg = str(cfg.get("run_label") or "").strip()
    if run_label_cfg:
        return run_label_cfg

    if bool(cfg.get("slurm_run_label", True)):
        slurm_label = _slurm_run_label()
        if slurm_label:
            return slurm_label

    return None


def run_eval(cfg: DictConfig) -> dict[str, Path]:
    """Evaluate each model across all tasks using experiment config."""
    data_root = Path(cfg.data_root)
    if not data_root.is_absolute():
        data_root = Path.cwd() / data_root

    raw_version = cfg.get("version", "current")
    if str(raw_version).strip().lower() == "current":
        version = detect_data_version(data_root)
    else:
        version = str(raw_version)

    output_base = Path(cfg.get("output_dir", "results"))
    if not output_base.is_absolute():
        output_base = Path.cwd() / output_base
    output_base = _normalize_output_base(output_base, version)

    device = resolve_device(str(cfg.get("device", "auto")))
    batch_size = max(1, int(cfg.get("batch_size", 1)))
    num_runs = max(1, int(cfg.get("num_runs", 1)))
    true_random_option_order = bool(cfg.get("true_random_option_order", False))
    task_overrides_cfg = cfg.get("task_overrides") or {}
    task_overrides = OmegaConf.to_container(task_overrides_cfg, resolve=True) if isinstance(task_overrides_cfg, DictConfig) else task_overrides_cfg
    if not isinstance(task_overrides, dict):
        task_overrides = {}
    lang_suffix = _results_language_suffix(task_overrides)
    prompt_language = _prompt_language(task_overrides)
    if not true_random_option_order and num_runs > 1:
        print(
            "  num_runs > 1 requested without true_random_option_order; "
            "running once because deterministic ordering is repeatable."
        )
        num_runs = 1
    results = {}

    for model_entry in cfg.models:
        # Support both string ("smolvlm2") and dict ({"name": "smolvlm2", "size": "2.2B"})
        if isinstance(model_entry, str):
            model_name = model_entry
            model_overrides = {}
        else:
            model_entry = OmegaConf.to_container(model_entry, resolve=True)
            model_name = model_entry.pop("name")
            model_overrides = model_entry

        base_cfg = load_model_config(model_name)
        if base_cfg is None:
            print(f"  Skip model {model_name}: no config found", file=sys.stderr)
            continue

        model_cfg = resolve_model_config(
            model_name=model_name,
            model_overrides=model_overrides,
            model_config=base_cfg,
        )

        size = str(model_cfg.get("size", "")).strip()
        model_label = f"{model_name}-{size}" if size else model_name
        model_label = f"{model_label}{lang_suffix}"

        # Load model once for all tasks
        try:
            model = build_model(
                model_name=model_name,
                model_cfg=model_cfg,
                device=device,
                auto_load=True,
            )
        except ValueError as exc:
            print(f"  Skip model {model_name}: {exc}", file=sys.stderr)
            continue

        model_base_dir = output_base / version / model_label

        for run_index in range(1, num_runs + 1):
            run_seed = (
                random.SystemRandom().getrandbits(63)
                if true_random_option_order
                else None
            )
            run_group = _resolve_run_label(
                cfg=cfg,
                true_random_option_order=true_random_option_order,
            )
            run_subdir = f"{run_index:04d}"
            model_dir = (
                (
                    model_base_dir / run_group / run_subdir
                    if run_group
                    else model_base_dir / run_subdir
                )
                if true_random_option_order
                else model_base_dir
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "dataset_version": version,
                "model": model_name,
                "model_size": size,
                "model_label": model_label,
                "prompt_language": prompt_language,
                "device": device,
                "batch_size": batch_size,
                "num_runs": num_runs,
                "run_index": run_index,
                "run_group": run_group,
                "run_subdir": run_subdir if true_random_option_order else None,
                "true_random_option_order": true_random_option_order,
                "run_seed": run_seed,
                "tasks": [str(t) for t in cfg.tasks],
            }
            (model_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            cache_path = model_dir / "cache" / "responses.json"
            cache = load_cache(cache_path)
            task_accuracies = {}
            run_desc = (
                f" [{run_group}/{run_subdir}]"
                if (true_random_option_order and run_group)
                else (f" [{run_subdir}]" if true_random_option_order else "")
            )

            for task_id in cfg.tasks:
                task_cfg = load_task_config(task_id)
                if task_cfg is None:
                    print(f"  Skip task {task_id}: no config found", file=sys.stderr)
                    continue

                # Check model capabilities vs task context_type
                capabilities = model_cfg.get("capabilities", [])
                context_type = task_cfg.get("context_type", "none")
                if capabilities and context_type not in capabilities and context_type != "none":
                    print(f"  Skip {task_id}: model {model_name} lacks capability '{context_type}'", file=sys.stderr)
                    continue

                # Load task dataset
                global_overrides = task_overrides.get("__all__", {}) if isinstance(task_overrides, dict) else {}
                task_specific_overrides = task_overrides.get(task_id, {}) if isinstance(task_overrides, dict) else {}
                overrides = {}
                if isinstance(global_overrides, dict):
                    overrides.update(global_overrides)
                if isinstance(task_specific_overrides, dict):
                    overrides.update(task_specific_overrides)
                overrides["true_random_option_order"] = true_random_option_order
                overrides["option_order_run_seed"] = run_seed
                task_def = get_task_def(task_id, version, data_root=data_root, task_overrides=overrides)
                if task_def is None:
                    print(f"  Skip {task_id}: no task def for version={version}", file=sys.stderr)
                    continue

                dataset_cls = get_task_dataset(task_id)
                if dataset_cls is None:
                    print(f"  Skip {task_id}: no dataset registered", file=sys.stderr)
                    continue

                dataset = dataset_cls(task_def=task_def, version=version, data_root=data_root)
                if len(dataset) == 0:
                    print(f"  Skip {task_id}: empty dataset", file=sys.stderr)
                    continue

                # Load human proportions if available
                human_props: dict = {}
                if task_def.human_response_path and Path(task_def.human_response_path).exists():
                    human_props = load_human_proportions(task_def.human_response_path)
                    if human_props:
                        print(
                            f"  {task_id}: loaded human proportions for "
                            f"{len(human_props)} items"
                        )

                # Evaluate each trial
                task_results = []
                task_trials = []
                max_new_tokens = model_cfg.get("max_new_tokens", 64)

                for chunk_start in tqdm(
                    range(0, len(dataset), batch_size),
                    desc=f"  {task_id}{run_desc}",
                    unit="batch",
                ):
                    chunk_trials: list[dict] = []
                    chunk_hashes: list[str] = []
                    chunk_results: list[dict | None] = []
                    uncached_positions: list[int] = []
                    uncached_trials: list[dict] = []

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
                                f"Model {model_name} returned {len(uncached_results)} "
                                f"results for {len(uncached_trials)} trials in batch."
                            )
                        for pos, result in zip(uncached_positions, uncached_results):
                            trial_for_result = chunk_trials[pos]
                            result["option_order_seed"] = str(
                                trial_for_result.get("option_order_seed", "")
                            )
                            # Annotate with human comparison metrics when data is available
                            if human_props:
                                annotate_human_metrics(
                                    result, human_props.get(result["item_uid"])
                                )
                            h = chunk_hashes[pos]
                            cache[h] = result
                            save_cache(cache_path, cache)
                            chunk_results[pos] = result

                    task_results.extend([r for r in chunk_results if r is not None])

                # Write per-task CSV
                write_task_csv(model_dir, task_id, task_results)
                write_task_npy(model_dir, task_id, task_results, task_trials=task_trials)
                post_paths = postprocess_task_outputs(
                    task_id=task_id,
                    model_dir=model_dir,
                    task_results=task_results,
                    task_trials=task_trials,
                )
                for out_path in post_paths:
                    print(f"  {model_name}/{task_id}{run_desc}: wrote {out_path}")

                # Compute accuracy
                correct = sum(1 for r in task_results if r["is_correct"])
                task_accuracies[task_id] = correct / len(task_results) if task_results else 0.0
                print(
                    f"  {model_name}/{task_id}{run_desc}: "
                    f"{task_accuracies[task_id]:.4f} ({correct}/{len(task_results)})"
                )

            # Write cross-task summary
            summary_path = write_summary_csv(model_dir, task_accuracies)
            result_key = (
                (
                    f"{model_name}:{run_group}:{run_subdir}"
                    if run_group
                    else f"{model_name}:{run_subdir}"
                )
                if true_random_option_order
                else model_name
            )
            results[result_key] = summary_path
            print(f"  Summary{run_desc}: {summary_path}")

    return results
