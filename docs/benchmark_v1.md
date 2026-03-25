# Benchmark v1 Protocol

This document defines the reproducible `v1` benchmark protocol for this repository.

## Goal

Provide a stable, repeatable benchmark bundle with:

- fixed task protocols,
- explicit randomness handling,
- chance-aware reporting, and
- machine-readable outputs for tracking progress.

## Task Protocols

### Math (EGMA)

- Corpus: `data/assets/<version>/corpus/egma-math/test-combined-math-cat.csv`
- Prompt builder: `scripts/build_math_prompts.py`
- Default setting: Number Line hint uses `coarse` (script default)
- Evaluator: `scripts/run_smolvlmv2_math_eval.py`
- Model default: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- Output metrics:
  - overall accuracy,
  - parse rate,
  - by-problem-type accuracy vs guess baseline.

### Theory of Mind

- Corpus: `data/assets/<version>/corpus/theory-of-mind/theory-of-mind-item-bank.csv`
- Evaluator: `scripts/run_smolvlmv2_tom_eval.py`
- Official v1 comparison: robust multi-seed runs via `scripts/run_tom_robustness.py`
- Default variants:
  - `none` (stateless reference),
  - `state_model` (character-memory stateful model mode).
- Required randomization: shuffled options with multiple seeds.
- Required reporting:
  - mean and std across seeds,
  - chance baseline and lift vs chance.

## Robustness Settings

- ToM seeds: `1,2,3,4,5` by default.
- Report mean, std, min, max for each variant.
- Avoid single-run headlines.

## Benchmark Bundle Layout

Each run writes to:

`results/benchmark/v1/<timestamp>/`

With:

- `run_metadata.json` - model id, data version, seeds, variants, device, commit hash.
- `benchmark_summary.csv` - compact top-line metrics.
- `math/` - prompt, predictions, summary, by-type CSV/plot.
- `tom/` - per-run and aggregate robustness files.

## Notes

- Chance-aware comparison is mandatory for ToM because option counts vary by item.
- If protocol details change, bump to `v2` rather than overwriting `v1`.
