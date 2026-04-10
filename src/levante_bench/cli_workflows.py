"""Shared CLI workflow helpers and benchmark presets."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_SMOLVLM2_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

WORKFLOW_SCRIPTS = {
    "benchmark-v1": "run_benchmark_v1.py",
    "shapebias-side-bias": "analyze_shapebias_side_bias.py",
    "shapebias-bias-decomposition": "run_shapebias_bias_decomposition.py",
    "shapebias-validity": "compute_shapebias_validity.py",
    "smol-math": "run_smolvlmv2_math_eval.py",
    "smol-tom": "run_smolvlmv2_tom_eval.py",
    "smol-vocab": "run_smolvlmv2_vocab_eval.py",
    "tom-modal": "run_tom_modal_eval.py",
    "tom-robustness": "run_tom_robustness.py",
}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_default_data_version(data_root: Path | None = None) -> str:
    """Return the most-recent asset version, or the LEVANTE_DATA_VERSION env var."""
    from levante_bench.config.defaults import detect_data_version
    return detect_data_version(data_root or project_root() / "data")


def normalize_passthrough(args: list[str] | None) -> list[str]:
    out = list(args or [])
    if out and out[0] == "--":
        out = out[1:]
    return out


def run_command(cmd: list[str], cwd: Path) -> int:
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def workflow_script_path(root: Path, workflow: str) -> Path:
    return root / "scripts" / WORKFLOW_SCRIPTS[workflow]


def workflow_command(root: Path, workflow: str, passthrough: list[str] | None) -> list[str]:
    script_path = workflow_script_path(root, workflow)
    return [sys.executable, str(script_path), *normalize_passthrough(passthrough)]


def benchmark_command(
    root: Path,
    benchmark: str,
    data_version: str,
    model_id: str,
    device: str,
    max_items_math: int | None = None,
    max_items_tom: int | None = None,
    max_items_vocab: int | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    if benchmark == "v1":
        cmd = [
            sys.executable,
            str(root / "scripts" / "run_benchmark_v1.py"),
            "--data-version",
            data_version,
            "--device",
            device,
            "--model-id",
            model_id,
        ]
        if max_items_math is not None:
            cmd.extend(["--max-items-math", str(max_items_math)])
        if max_items_tom is not None:
            cmd.extend(["--max-items-tom", str(max_items_tom)])
    elif benchmark == "vocab":
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = root / "results" / "benchmark" / "vocab" / ts
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(root / "scripts" / "run_smolvlmv2_vocab_eval.py"),
            "--corpus-csv",
            str(root / "data" / "assets" / data_version / "corpus" / "vocab" / "vocab-item-bank.csv"),
            "--visual-dir",
            str(root / "data" / "assets" / data_version / "visual" / "vocab"),
            "--output-jsonl",
            str(out_dir / "vocab-preds.jsonl"),
            "--summary-json",
            str(out_dir / "vocab-summary.json"),
            "--composite-dir",
            str(out_dir / "vocab-composites"),
            "--device",
            device,
            "--model-id",
            model_id,
        ]
        if max_items_vocab is not None:
            cmd.extend(["--max-items", str(max_items_vocab)])
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    cmd.extend(normalize_passthrough(extra_args))
    return cmd
