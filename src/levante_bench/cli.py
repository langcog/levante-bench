"""CLI: run-eval, run-benchmark, run-workflow, run-comparison."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from levante_bench.cli_workflows import (
    DEFAULT_SMOLVLM2_MODEL,
    WORKFLOW_SCRIPTS,
    benchmark_command,
    get_default_data_version,
    normalize_passthrough,
    project_root,
    run_command,
    workflow_command,
    workflow_script_path,
)
from levante_bench.runtime import load_model, run_trials


def _load_local_env() -> None:
    """Load KEY=VALUE pairs from repo-local .env without overriding exports."""
    env_path = _project_root() / ".env"
    if not env_path.exists() or not env_path.is_file():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (
            len(value) >= 2
            and ((value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")))
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)

    # Hugging Face libraries commonly accept either env name; keep them aligned.
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)


def _project_root() -> Path:
    return project_root()


def _run_experiment_style_args(cli_args: list[str]) -> int:
    """Compatibility path for eval-style CLI args: experiment=... overrides."""
    from omegaconf import OmegaConf

    from levante_bench.config.loader import (
        load_experiment,
        load_model_config,
        load_task_config,
    )
    from levante_bench.evaluation.runner import resolve_device

    experiment_path: str | None = None
    overrides: list[str] = []
    for arg in cli_args:
        if arg.startswith("experiment="):
            experiment_path = arg.split("=", 1)[1]
        else:
            overrides.append(arg)

    if not experiment_path:
        print("Required: experiment=<path_to_config.yaml>", file=sys.stderr)
        return 1

    cfg = load_experiment(experiment_path=experiment_path, cli_overrides=overrides)

    # If models are dicts (new format), go directly to run_eval
    models_raw = cfg.get("models") or []
    has_dict_models = any(not isinstance(m, str) for m in models_raw)
    if has_dict_models:
        from levante_bench.evaluation.runner import run_eval
        results = run_eval(cfg)
        if not results:
            print("No results produced.", file=sys.stderr)
            return 1
        for model_id, path in results.items():
            print(f"  {model_id}: {path}")
        return 0

    tasks = [str(t) for t in (cfg.get("tasks") or [])]
    models = [str(m) for m in models_raw]
    version = str(cfg.get("version") or "current")
    device = resolve_device(str(cfg.get("device") or "auto"))
    output_dir = str(cfg.get("output_dir") or "results")
    root = _project_root()

    if not tasks:
        print("No tasks configured in experiment YAML.", file=sys.stderr)
        return 1
    if not models:
        print("No models configured in experiment YAML.", file=sys.stderr)
        return 1

    for task_id in tasks:
        if load_task_config(task_id) is None:
            print(f"No task config found for '{task_id}' in configs/tasks.", file=sys.stderr)
            return 1

    resolved_models: dict[str, dict] = {}
    for model_name in models:
        model_cfg = load_model_config(model_name)
        if model_cfg is None:
            print(f"No model config found for '{model_name}' in configs/models.", file=sys.stderr)
            return 1
        resolved_models[model_name] = OmegaConf.to_container(model_cfg, resolve=True)

    # For vocab-only experiments, route through benchmark vocab so hf_name from YAML is used.
    if tasks == ["vocab"]:
        exit_code = 0
        max_items_vocab = cfg.get("max_items_vocab")
        for _, model_cfg in resolved_models.items():
            hf_name = model_cfg.get("hf_name")
            if not hf_name:
                print("Missing hf_name in model config for vocab experiment.", file=sys.stderr)
                return 1
            cmd = benchmark_command(
                root=root,
                benchmark="vocab",
                data_version=version,
                model_id=str(hf_name),
                device=device,
                max_items_vocab=int(max_items_vocab) if max_items_vocab is not None else None,
                extra_args=[],
            )
            print("Running experiment:", " ".join(cmd))
            exit_code = exit_code or run_command(cmd, cwd=root)
        return exit_code

    # For SmolVLM task workflows, route to dedicated script pipelines and use YAML hf_name.
    smol_tasks = {"egma-math", "theory-of-mind"}
    if set(tasks).issubset(smol_tasks):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_root = Path(output_dir) / "experiments" / version / timestamp
        out_root.mkdir(parents=True, exist_ok=True)
        exit_code = 0
        max_items_math = cfg.get("max_items_math")
        max_items_tom = cfg.get("max_items_tom")
        for _, model_cfg in resolved_models.items():
            hf_name = model_cfg.get("hf_name")
            if not hf_name:
                print("Missing hf_name in model config for SmolVLM experiment.", file=sys.stderr)
                return 1
            if "egma-math" in tasks:
                math_dir = out_root / "math"
                math_dir.mkdir(parents=True, exist_ok=True)
                prompts = math_dir / "egma-math-prompts.jsonl"
                prompts_eval = math_dir / "egma-math-prompts-eval.jsonl"
                preds = math_dir / "egma-math-preds.jsonl"
                summary = math_dir / "egma-math-summary.json"
                by_type_csv = math_dir / "egma-math-by-type.csv"
                by_type_png = math_dir / "egma-math-by-type.png"
                math_corpus = root / "data" / "assets" / version / "corpus" / "egma-math" / "math-item-bank.csv"
                numberline_graphics = root / "local_data" / "numberline-graphics" / "egma-math"
                cmd_build = [
                    sys.executable,
                    str(root / "scripts" / "build_math_prompts.py"),
                    "--corpus-csv",
                    str(math_corpus),
                    "--output",
                    str(prompts),
                    "--numberline-graphics-dir",
                    str(numberline_graphics),
                    "--numberline-hint",
                    "none",
                    "--numberline-instruction-style",
                    "minimal",
                ]
                cmd_eval = [
                    sys.executable,
                    str(root / "scripts" / "run_smolvlmv2_math_eval.py"),
                    "--input-jsonl",
                    str(prompts_eval),
                    "--output-jsonl",
                    str(preds),
                    "--summary-json",
                    str(summary),
                    "--model-id",
                    str(hf_name),
                    "--device",
                    device,
                ]
                if max_items_math is not None:
                    cmd_eval.extend(["--max-items", str(int(max_items_math))])
                cmd_analyze = [
                    sys.executable,
                    str(root / "scripts" / "analyze_math_type_accuracy.py"),
                    "--prompts-jsonl",
                    str(prompts),
                    "--preds-jsonl",
                    str(preds),
                    "--output-csv",
                    str(by_type_csv),
                    "--output-png",
                    str(by_type_png),
                ]
                print("Running experiment:", " ".join(cmd_build))
                exit_code = exit_code or run_command(cmd_build, cwd=root)
                if exit_code:
                    continue
                if max_items_math is not None:
                    prompt_lines = [
                        line
                        for line in prompts.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    prompt_lines = prompt_lines[: int(max_items_math)]
                    prompts_eval.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
                else:
                    prompts_eval = prompts
                print("Running experiment:", " ".join(cmd_eval))
                exit_code = exit_code or run_command(cmd_eval, cwd=root)
                if exit_code:
                    continue
                print("Running experiment:", " ".join(cmd_analyze))
                cmd_analyze[3] = str(prompts_eval)
                exit_code = exit_code or run_command(cmd_analyze, cwd=root)
            if "theory-of-mind" in tasks:
                tom_dir = out_root / "tom"
                tom_dir.mkdir(parents=True, exist_ok=True)
                tom_corpus = root / "data" / "assets" / version / "corpus" / "theory-of-mind" / "theory-of-mind-item-bank.csv"
                preds = tom_dir / "tom-preds.jsonl"
                summary = tom_dir / "tom-summary.json"
                cmd_tom = [
                    sys.executable,
                    str(root / "scripts" / "run_smolvlmv2_tom_eval.py"),
                    "--corpus-csv",
                    str(tom_corpus),
                    "--output-jsonl",
                    str(preds),
                    "--summary-json",
                    str(summary),
                    "--model-id",
                    str(hf_name),
                    "--device",
                    device,
                ]
                if max_items_tom is not None:
                    cmd_tom.extend(["--max-items", str(int(max_items_tom))])
                print("Running experiment:", " ".join(cmd_tom))
                exit_code = exit_code or run_command(cmd_tom, cwd=root)
        return exit_code

    print(
        "Experiment YAML fallback: non-vocab tasks use run-eval with model IDs from configs.models.",
        file=sys.stderr,
    )
    run_eval_ns = argparse.Namespace(
        task=tasks,
        model=models,
        version=version,
        device=device,
        output_dir=output_dir,
        batch_size=int(cfg.get("batch_size", 1)),
        include_numberline=bool(cfg.get("include_numberline", False)),
        prompt_language=str(cfg.get("prompt_language", "en")),
        num_runs=int(cfg.get("num_runs", 1)),
        true_random_option_order=bool(cfg.get("true_random_option_order", False)),
        run_label=str(cfg.get("run_label", "") or ""),
        slurm_run_label=bool(cfg.get("slurm_run_label", True)),
    )
    return cmd_run_eval(run_eval_ns)


def cmd_list_tasks(_: argparse.Namespace) -> int:
    from levante_bench.config import list_tasks
    for t in list_tasks():
        print(t)
    return 0


def cmd_list_models(_: argparse.Namespace) -> int:
    from levante_bench.models import list_models
    for m in list_models():
        print(m)
    return 0


def cmd_run_eval(args: argparse.Namespace) -> int:
    from omegaconf import OmegaConf

    from levante_bench.config import list_tasks
    from levante_bench.evaluation.runner import resolve_device, run_eval
    from levante_bench.models import list_models

    task_ids = args.task if args.task else None
    model_ids = args.model if args.model else None
    from levante_bench.config.defaults import detect_data_version

    version_arg = args.version or "current"
    if str(version_arg).strip().lower() == "current":
        version = detect_data_version(_project_root() / "data")
    else:
        version = str(version_arg)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else _project_root() / "results"
    data_root = _project_root() / "data"
    print(
        f"Running evaluation: version={version}, device={device}, "
        f"output_base={output_dir} (per-model: {output_dir / version}/<model>-<size>[-<lang>]/)"
    )
    print(f"  Data root: {data_root}")
    if task_ids:
        print(f"  Tasks: {', '.join(task_ids)}")
    if model_ids:
        print(f"  Models: {', '.join(model_ids)}")
    if args.include_numberline:
        print("  Egma-math override: include Number Line items")
    if args.prompt_language and args.prompt_language != "en":
        print(f"  Prompt language override: {args.prompt_language}")
    if int(args.num_runs) > 1:
        print(f"  num_runs: {int(args.num_runs)}")
    if bool(args.true_random_option_order):
        print("  Option ordering mode: true_random")
    if str(args.run_label or "").strip():
        print(f"  Run group label: {str(args.run_label).strip()}")
    elif bool(args.true_random_option_order) and bool(args.slurm_run_label):
        print("  Run groups: Slurm-aware (job/task IDs when available)")

    cfg = OmegaConf.create(
        {
            "tasks": task_ids or list_tasks(),
            "models": model_ids or list_models(),
            "version": version,
            "device": device,
            "batch_size": int(args.batch_size),
            "num_runs": int(args.num_runs),
            "true_random_option_order": bool(args.true_random_option_order),
            "run_label": str(args.run_label or ""),
            "slurm_run_label": bool(args.slurm_run_label),
            "output_dir": str(output_dir),
            "data_root": str(data_root),
            "task_overrides": {
                "__all__": {"prompt_language": str(args.prompt_language or "en")},
                "egma-math": {"include_numberline": bool(args.include_numberline)},
            },
        }
    )

    results = run_eval(cfg)
    if not results:
        print("No outputs written. Check that data/responses/<version>/ and data/assets/<version>/ exist and item_uid index matches trials.", file=sys.stderr)
        return 1
    print(f"Success: wrote {len(results)} file(s)")
    for model_id, path in results.items():
        print(f"  {model_id}\t{path}")
    return 0


def cmd_check_gpu(_: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("torch is not installed in the active environment.", file=sys.stderr)
        return 1
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    print(f"cuda_available={available}")
    print(f"gpu_count={count}")
    for i in range(count):
        print(f"gpu[{i}]={torch.cuda.get_device_name(i)}")
    return 0


def cmd_run_workflow(args: argparse.Namespace) -> int:
    root = _project_root()
    script_path = workflow_script_path(root, args.workflow)
    if not script_path.exists():
        print(f"Workflow script not found: {script_path}", file=sys.stderr)
        return 1
    cmd = workflow_command(root, args.workflow, args.script_args)
    print("Running workflow:", " ".join(cmd))
    return run_command(cmd, cwd=root)


def cmd_run_benchmark(args: argparse.Namespace) -> int:
    from levante_bench.evaluation.runner import resolve_device

    root = _project_root()
    device = resolve_device(args.device)
    data_version = args.data_version or get_default_data_version(root / "data")
    model_id = args.model_id or DEFAULT_SMOLVLM2_MODEL

    try:
        cmd = benchmark_command(
            root=root,
            benchmark=args.benchmark,
            data_version=data_version,
            model_id=model_id,
            device=device,
            max_items_math=args.max_items_math,
            max_items_tom=args.max_items_tom,
            max_items_vocab=args.max_items_vocab,
            extra_args=normalize_passthrough(args.benchmark_args),
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    print("Running benchmark:", " ".join(cmd))
    return run_command(cmd, cwd=root)


def cmd_run_comparison(args: argparse.Namespace) -> int:
    root = _project_root()
    script = root / "comparison" / "compare_levante.R"
    if not script.exists():
        print("comparison/compare_levante.R not found", file=sys.stderr)
        return 1
    if not args.task or not args.model:
        print("run-comparison requires --task and --model", file=sys.stderr)
        return 1
    cmd = [
        "Rscript",
        str(script),
        "--task", args.task,
        "--model", args.model,
        "--version", args.version or "current",
        "--results-dir", args.results_dir or "results",
        "--project-root", str(root),
        "--output-dir", str(root / (args.output_dir or "results/comparison")),
    ]
    if getattr(args, "output_dkl", None):
        cmd.extend(["--output-dkl", args.output_dkl])
    if getattr(args, "output_accuracy", None):
        cmd.extend(["--output-accuracy", args.output_accuracy])
    return run_command(cmd, cwd=root)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_no}: expected JSON object per line, got {type(value).__name__}"
                )
            rows.append(value)
    return rows


def _validate_trials(trials: list[dict], source_path: Path) -> None:
    required = ("trial_id", "item_uid", "prompt", "option_labels")
    for idx, trial in enumerate(trials, start=1):
        missing = [key for key in required if key not in trial]
        if missing:
            raise ValueError(
                f"{source_path}:{idx}: missing required trial field(s): {', '.join(missing)}"
            )
        if not isinstance(trial["option_labels"], list) or not trial["option_labels"]:
            raise ValueError(
                f"{source_path}:{idx}: option_labels must be a non-empty list"
            )
        if "answer_format" not in trial and "correct_label" not in trial and "target_value" not in trial:
            raise ValueError(
                f"{source_path}:{idx}: include one of correct_label, target_value, or answer_format"
            )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def cmd_run_trials_jsonl(args: argparse.Namespace) -> int:
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    if not input_path.exists():
        print(f"Input JSONL not found: {input_path}", file=sys.stderr)
        return 1
    if not input_path.is_file():
        print(f"Input path is not a file: {input_path}", file=sys.stderr)
        return 1
    if not args.model and not args.model_config_path:
        print(
            "run-trials-jsonl requires --model or --model-config-path",
            file=sys.stderr,
        )
        return 1

    try:
        trials = _read_jsonl(input_path)
        _validate_trials(trials, input_path)
        model = load_model(
            model_name=args.model,
            model_config_path=args.model_config_path,
            configs_root=args.configs_root,
            device=args.device,
            auto_load=True,
        )
        results = run_trials(
            model=model,
            trials=trials,
            max_new_tokens=args.max_new_tokens,
            task_id=args.task_id,
        )
        _write_jsonl(output_path, results)
    except Exception as exc:
        print(f"run-trials-jsonl failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Success: wrote {len(results)} result row(s) to {output_path}",
        file=sys.stderr,
    )
    return 0


def add_list_tasks_parser(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("list-tasks", help="List registered task IDs")


def add_list_models_parser(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("list-models", help="List registered model IDs")


def add_check_gpu_parser(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("check-gpu", help="Report local CUDA/GPU availability")


def add_run_eval_parser(sub: argparse._SubParsersAction) -> None:
    pe = sub.add_parser("run-eval", help="Run evaluation (write .npy per task/model)")
    pe.add_argument("--task", action="append", help="Task ID (repeat for multiple)")
    pe.add_argument("--model", action="append", help="Model ID (repeat for multiple)")
    pe.add_argument("--version", default="current", help="Data/asset version")
    pe.add_argument("--device", default="auto", help="Device for model: auto|cpu|cuda")
    pe.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Trials per model call (uses model batch API when implemented).",
    )
    pe.add_argument("--output-dir", help="Output directory (default: results/<version>)")
    pe.add_argument(
        "--include-numberline",
        action="store_true",
        help="For egma-math in runner path, include Number Line trial types from manifest.",
    )
    pe.add_argument(
        "--prompt-language",
        default="en",
        help="Prompt language code from translations CSV (e.g., en, de, es-CO).",
    )
    pe.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of evaluation runs (used for true-random option ordering).",
    )
    pe.add_argument(
        "--true-random-option-order",
        action="store_true",
        help=(
            "Use true-random option ordering. "
            "When enabled, outputs are written to per-run subfolders (0001, 0002, ...)."
        ),
    )
    pe.add_argument(
        "--run-label",
        default="",
        help=(
            "Optional parent folder label for true-random runs. "
            "Run subfolders remain sequential (0001, 0002, ...)."
        ),
    )
    pe.add_argument(
        "--no-slurm-run-label",
        action="store_false",
        dest="slurm_run_label",
        help="Disable automatic Slurm job/task run labels in true-random mode.",
    )
    pe.set_defaults(slurm_run_label=True)


def add_run_workflow_parser(sub: argparse._SubParsersAction) -> None:
    pw = sub.add_parser("run-workflow", help="Run integrated benchmark/test workflow scripts")
    pw.add_argument("--workflow", required=True, choices=sorted(WORKFLOW_SCRIPTS.keys()))
    pw.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the workflow script (prefix with --).",
    )


def add_run_benchmark_parser(sub: argparse._SubParsersAction) -> None:
    pb = sub.add_parser("run-benchmark", help="Run integrated benchmark presets (v1, vocab)")
    pb.add_argument("--benchmark", required=True, choices=["v1", "vocab"])
    pb.add_argument("--data-version", default=None, help="Data/assets version (default: auto-detect from data/assets/)")
    pb.add_argument("--model-id", default=DEFAULT_SMOLVLM2_MODEL, help="Model id")
    pb.add_argument("--device", default="auto", help="Device: auto|cpu|cuda")
    pb.add_argument("--max-items-math", type=int, default=None, help="Optional cap for v1 math")
    pb.add_argument("--max-items-tom", type=int, default=None, help="Optional cap for v1 ToM")
    pb.add_argument("--max-items-vocab", type=int, default=None, help="Optional cap for vocab benchmark")
    pb.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to the underlying benchmark script (prefix with --).",
    )


def add_run_comparison_parser(sub: argparse._SubParsersAction) -> None:
    pc = sub.add_parser("run-comparison", help="Run R comparison (D_KL by age+item_uid, accuracy by item_uid)")
    pc.add_argument("--task", required=True, help="Task ID")
    pc.add_argument("--model", required=True, help="Model ID")
    pc.add_argument("--version", default="current", help="Data version")
    pc.add_argument("--results-dir", default="results", help="Results directory name")
    pc.add_argument("--output-dir", default="results/comparison", help="Directory for D_KL and accuracy CSVs")
    pc.add_argument("--output-dkl", help="Output path for D_KL CSV (default: <output-dir>/<task>_<model>_d_kl.csv)")
    pc.add_argument("--output-accuracy", help="Output path for accuracy CSV (default: <output-dir>/<task>_<model>_accuracy.csv)")


def add_run_trials_jsonl_parser(sub: argparse._SubParsersAction) -> None:
    pr = sub.add_parser("run-trials-jsonl", help="Run external trial JSONL through a model and write result JSONL")
    pr.add_argument("--input-jsonl", required=True, help="Input JSONL path; one trial object per line")
    pr.add_argument("--output-jsonl", required=True, help="Output JSONL path for evaluation results")
    pr.add_argument("--model", help="Registered model ID (e.g. qwen35)")
    pr.add_argument("--model-config-path", help="Path to model config YAML (can provide name/hf_name)")
    pr.add_argument("--configs-root", help="Optional configs root for model lookup (contains models/*.yaml)")
    pr.add_argument("--device", default="auto", help="Device for model: auto|cpu|cuda")
    pr.add_argument("--max-new-tokens", type=int, default=None, help="Default max_new_tokens when trial field is missing")
    pr.add_argument("--task-id", default=None, help="Optional task_id to inject when missing in trial rows")


def main() -> int:
    _load_local_env()

    # Eval-branch compatibility mode:
    # python -m levante_bench.cli experiment=configs/experiments/experiment.yaml device=cuda
    raw_args = sys.argv[1:]
    if any(arg.startswith("experiment=") for arg in raw_args):
        return _run_experiment_style_args(raw_args)

    parser = argparse.ArgumentParser(prog="levante-bench", description="LEVANTE VLM benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    add_list_tasks_parser(sub)
    add_list_models_parser(sub)
    add_check_gpu_parser(sub)
    add_run_eval_parser(sub)
    add_run_workflow_parser(sub)
    add_run_benchmark_parser(sub)
    add_run_comparison_parser(sub)
    add_run_trials_jsonl_parser(sub)
    args = parser.parse_args()
    if args.command == "list-tasks":
        return cmd_list_tasks(args)
    if args.command == "list-models":
        return cmd_list_models(args)
    if args.command == "check-gpu":
        return cmd_check_gpu(args)
    if args.command == "run-eval":
        return cmd_run_eval(args)
    if args.command == "run-workflow":
        return cmd_run_workflow(args)
    if args.command == "run-benchmark":
        return cmd_run_benchmark(args)
    if args.command == "run-comparison":
        return cmd_run_comparison(args)
    if args.command == "run-trials-jsonl":
        return cmd_run_trials_jsonl(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
