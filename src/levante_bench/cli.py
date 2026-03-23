"""CLI: run-eval, list-tasks, list-models, run-comparison."""

import argparse
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


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
    from levante_bench.evaluation.runner import run_eval
    task_ids = args.task if args.task else None
    model_ids = args.model if args.model else None
    version = args.version or "current"
    device = args.device or "cpu"
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        output_dir = _project_root() / "results" / version
    data_root = _project_root() / "data"
    print(f"Running evaluation: version={version}, device={device}, output={output_dir}")
    print(f"  Data root: {data_root}")
    if task_ids:
        print(f"  Tasks: {', '.join(task_ids)}")
    if model_ids:
        print(f"  Models: {', '.join(model_ids)}")
    results = run_eval(
        task_ids=task_ids,
        model_ids=model_ids,
        version=version,
        device=device,
        output_dir=output_dir,
        data_root=data_root,
    )
    if not results:
        print("No outputs written. Check that data/raw/<version>/ and data/assets/<version>/ exist and item_uid index matches trials.", file=sys.stderr)
        return 1
    print(f"Success: wrote {len(results)} file(s)")
    for (task_id, model_id), path in results.items():
        print(f"  {task_id}\t{model_id}\t{path}")
    return 0


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
    r = subprocess.run(cmd, cwd=str(root))
    return r.returncode


def main() -> int:
    parser = argparse.ArgumentParser(prog="levante-bench", description="LEVANTE VLM benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    # list-tasks
    sub.add_parser("list-tasks", help="List registered task IDs")
    # list-models
    sub.add_parser("list-models", help="List registered model IDs")
    # run-eval
    pe = sub.add_parser("run-eval", help="Run evaluation (write .npy per task/model)")
    pe.add_argument("--task", action="append", help="Task ID (repeat for multiple)")
    pe.add_argument("--model", action="append", help="Model ID (repeat for multiple)")
    pe.add_argument("--version", default="current", help="Data/asset version")
    pe.add_argument("--device", default="cpu", help="Device for model")
    pe.add_argument("--output-dir", help="Output directory (default: results/<version>)")
    # run-comparison
    pc = sub.add_parser("run-comparison", help="Run R comparison (D_KL by age+item_uid, accuracy by item_uid)")
    pc.add_argument("--task", required=True, help="Task ID")
    pc.add_argument("--model", required=True, help="Model ID")
    pc.add_argument("--version", default="current", help="Data version")
    pc.add_argument("--results-dir", default="results", help="Results directory name")
    pc.add_argument("--output-dir", default="results/comparison", help="Directory for D_KL and accuracy CSVs")
    pc.add_argument("--output-dkl", help="Output path for D_KL CSV (default: <output-dir>/<task>_<model>_d_kl.csv)")
    pc.add_argument("--output-accuracy", help="Output path for accuracy CSV (default: <output-dir>/<task>_<model>_accuracy.csv)")
    args = parser.parse_args()
    if args.command == "list-tasks":
        return cmd_list_tasks(args)
    if args.command == "list-models":
        return cmd_list_models(args)
    if args.command == "run-eval":
        return cmd_run_eval(args)
    if args.command == "run-comparison":
        return cmd_run_comparison(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
