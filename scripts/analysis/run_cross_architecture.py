#!/usr/bin/env python3
"""Run the cross-architecture experiment across all model families and tasks.

Usage
-----
    python scripts/run_cross_architecture.py [--device mps|cuda|cpu]

Results are written to results/cross_architecture/<model>/<version>/.
Each model/task pair produces a detailed CSV plus a per-model summary.csv.

Re-running is safe: completed trials are cached in
results/cross_architecture/<model>/<version>/cache/responses.json
and skipped automatically.
"""
import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))

    from levante_bench.config.defaults import detect_data_version
    from levante_bench.evaluation.runner import run_eval

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None,
                        help="Override device (mps | cuda | cpu). "
                             "Defaults to value in experiment config.")
    parser.add_argument("--config",
                        default=str(repo_root / "configs" / "experiments" / "cross_architecture.yaml"),
                        help="Path to experiment YAML config.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Run only these model names (subset of config).")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Run only these task IDs (subset of config).")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Resolve data_root relative to repo root if needed
    data_root = Path(str(cfg.data_root))
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    cfg.data_root = str(data_root)

    # Auto-detect version when null
    if not cfg.get("version"):
        cfg.version = detect_data_version(data_root)
        print(f"Auto-detected data version: {cfg.version}")

    # CLI overrides
    if args.device:
        cfg.device = args.device
    if args.models:
        cfg.models = args.models
    if args.tasks:
        cfg.tasks = args.tasks

    # Resolve output_dir relative to repo root
    output_dir = Path(str(cfg.output_dir))
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    cfg.output_dir = str(output_dir)

    print(f"Device      : {cfg.device}")
    print(f"Models      : {list(cfg.models)}")
    print(f"Tasks       : {list(cfg.tasks)}")
    print(f"Output dir  : {cfg.output_dir}")
    print()

    summaries = run_eval(cfg)
    print("\nCompleted. Summary files:")
    for model, path in summaries.items():
        print(f"  {model}: {path}")


if __name__ == "__main__":
    main()
