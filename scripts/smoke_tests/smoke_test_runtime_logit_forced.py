#!/usr/bin/env python3
"""Smoke test for runtime logit-forced 1-vs-2 scoring."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from levante_bench.runtime import load_model, run_logit_forced_12


def _solid(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (64, 64), color=color).save(path, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test runtime logit-forced scoring."
    )
    parser.add_argument("--model", default="qwen35", help="Model id to load.")
    parser.add_argument("--device", default="auto", help="Device auto|cpu|cuda.")
    parser.add_argument(
        "--swap-correct",
        action="store_true",
        help="Enable swap-corrected logit-forced scoring.",
    )
    args = parser.parse_args()

    with TemporaryDirectory(prefix="levante-logit-smoke-") as tmp_dir:
        p = Path(tmp_dir)
        ref = p / "ref.png"
        a = p / "a.png"
        b = p / "b.png"
        _solid(ref, (200, 30, 30))
        _solid(a, (30, 200, 30))
        _solid(b, (30, 30, 200))

        model = load_model(model_name=args.model, device=args.device)
        result = run_logit_forced_12(
            model=model,
            prompt_text="Which candidate matches the reference image better? Reply 1 or 2.",
            image_paths=[str(ref), str(a), str(b)],
            swap_correct=args.swap_correct,
        )

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
