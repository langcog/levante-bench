"""Run evaluation: for each model, evaluate all tasks, write results."""

import inspect
import json
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
from levante_bench.evaluation.outputs import write_task_csv, write_summary_csv
from levante_bench.models import get_model_class
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

    device = resolve_device(str(cfg.get("device", "auto")))
    task_overrides_cfg = cfg.get("task_overrides") or {}
    task_overrides = OmegaConf.to_container(task_overrides_cfg, resolve=True) if isinstance(task_overrides_cfg, DictConfig) else task_overrides_cfg
    if not isinstance(task_overrides, dict):
        task_overrides = {}
    lang_suffix = _results_language_suffix(task_overrides)
    prompt_language = _prompt_language(task_overrides)
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

        # Merge experiment overrides into base model config, then resolve
        model_cfg = OmegaConf.to_container(base_cfg, resolve=False)
        model_cfg.update(model_overrides)
        model_cfg = OmegaConf.to_container(OmegaConf.create(model_cfg), resolve=True)

        size = str(model_cfg.get("size", "")).strip()
        model_label = f"{model_name}-{size}" if size else model_name
        model_label = f"{model_label}{lang_suffix}"

        # Load model once for all tasks
        model_cls = get_model_class(model_name)
        if model_cls is None:
            print(f"  Skip model {model_name}: not registered", file=sys.stderr)
            continue

        ctor_cfg = {
            k: v
            for k, v in model_cfg.items()
            if k
            not in {
                "name",
                "hf_name",
                "size",
                "max_new_tokens",
                "use_json_format",
                "capabilities",
            }
        }
        sig = inspect.signature(model_cls.__init__)
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not has_var_kwargs:
            accepted = {
                name
                for name in sig.parameters
                if name not in {"self", "model_name", "device"}
            }
            ctor_cfg = {k: v for k, v in ctor_cfg.items() if k in accepted}

        model = model_cls(model_name=model_cfg["hf_name"], device=device, **ctor_cfg)
        model.use_json_format = model_cfg.get("use_json_format", True)
        model.load()

        model_dir = output_base / version / model_label
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "dataset_version": version,
            "model": model_name,
            "model_size": size,
            "model_label": model_label,
            "prompt_language": prompt_language,
            "device": device,
            "tasks": [str(t) for t in cfg.tasks],
        }
        (model_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
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
            global_overrides = task_overrides.get("__all__", {}) if isinstance(task_overrides, dict) else {}
            task_specific_overrides = task_overrides.get(task_id, {}) if isinstance(task_overrides, dict) else {}
            overrides = {}
            if isinstance(global_overrides, dict):
                overrides.update(global_overrides)
            if isinstance(task_specific_overrides, dict):
                overrides.update(task_specific_overrides)
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

            for i in tqdm(range(len(dataset)), desc=f"  {task_id}", unit="trial"):
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
