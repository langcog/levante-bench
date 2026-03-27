"""Run evaluation: for each model, evaluate all tasks, write results."""

import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from levante_bench.config import get_task_def, load_model_config, load_task_config
from levante_bench.data.loaders import load_human_proportions
from levante_bench.evaluation.adapters import postprocess_task_outputs
from levante_bench.evaluation.cache import load_cache, save_cache, trial_hash
from levante_bench.evaluation.human_comparison import annotate_human_metrics
from levante_bench.evaluation.outputs import write_task_csv, write_summary_csv
from levante_bench.models import get_model_class
from levante_bench.tasks import get_task_dataset


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


def run_eval(cfg: DictConfig) -> dict[str, Path]:
    """Evaluate each model across all tasks using experiment config."""
    data_root = Path(cfg.data_root)
    version = cfg.get("version", "current")
    output_base = Path(cfg.get("output_dir", "results"))
    device = cfg.get("device", "cpu")
    results = {}

    for model_name in cfg.models:
        model_cfg = load_model_config(model_name)
        if model_cfg is None:
            print(f"  Skip model {model_name}: no config found", file=sys.stderr)
            continue

        # Resolve model config (handles ${size} interpolation etc.)
        model_cfg = OmegaConf.to_container(model_cfg, resolve=True)

        # Load model once for all tasks
        model_cls = get_model_class(model_name)
        if model_cls is None:
            print(f"  Skip model {model_name}: not registered", file=sys.stderr)
            continue

        model = model_cls(model_name=model_cfg["hf_name"], device=device)
        model.load()

        model_dir = output_base / model_name / version
        cache_path = model_dir / "cache" / "responses.json"
        cache = load_cache(cache_path)
        task_accuracies = {}

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
            task_def = get_task_def(task_id, version, data_root=data_root)
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

            for i in range(len(dataset)):
                trial = dataset[i]
                task_trials.append(trial)
                trial["max_new_tokens"] = max_new_tokens
                h = trial_hash(trial)

                if h in cache:
                    task_results.append(cache[h])
                    continue

                result = model.evaluate_trial(trial)

                # Annotate with human comparison metrics when data is available
                if human_props:
                    annotate_human_metrics(
                        result, human_props.get(result["item_uid"])
                    )

                cache[h] = result
                save_cache(cache_path, cache)
                task_results.append(result)

            # Write per-task CSV
            write_task_csv(model_dir, task_id, task_results)
            post_paths = postprocess_task_outputs(
                task_id=task_id,
                model_dir=model_dir,
                task_results=task_results,
                task_trials=task_trials,
            )
            for out_path in post_paths:
                print(f"  {model_name}/{task_id}: wrote {out_path}")

            # Compute accuracy
            correct = sum(1 for r in task_results if r["is_correct"])
            task_accuracies[task_id] = correct / len(task_results) if task_results else 0.0
            print(f"  {model_name}/{task_id}: {task_accuracies[task_id]:.4f} ({correct}/{len(task_results)})")

        # Write cross-task summary
        summary_path = write_summary_csv(model_dir, task_accuracies)
        results[model_name] = summary_path
        print(f"  Summary: {summary_path}")

    return results
