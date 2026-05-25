# Other Scoring Modes

This note documents ad hoc scoring-mode experiments that are not part of the default benchmark run.

## Forced Binary Label Scoring

Use `scripts/slurm/submit_forced_binary_all_tasks.sh` on Marlowe to rerun label-style tasks with forced binary scoring.

By default, it runs:

- Models: `cogvlm`, `smolvlm2:256M`, `smolvlm2:500M`
- Tasks: `egma-math`, `matrix-reasoning`, `mental-rotation`, `theory-of-mind`, `trog`, `vocab`
- Output root: `results/forced_binary_all_tasks`

Run defaults:

```bash
cd /projects/m000102/code/levante-bench
bash scripts/slurm/submit_forced_binary_all_tasks.sh
```

Dry run:

```bash
DRY_RUN=1 bash scripts/slurm/submit_forced_binary_all_tasks.sh
```

Run specific tasks:

```bash
TASKS_CSV="trog,vocab" \
bash scripts/slurm/submit_forced_binary_all_tasks.sh
```

Run specific models:

```bash
MODELS_CSV="cogvlm,smolvlm2:256M,qwen35:0.8B" \
bash scripts/slurm/submit_forced_binary_all_tasks.sh
```

You can also pass models as positional args:

```bash
bash scripts/slurm/submit_forced_binary_all_tasks.sh cogvlm smolvlm2:500M
```

## Environment Notes

The wrapper enables:

```bash
FORCE_BINARY_LABEL_SCORING=1
COGVLM_LABEL_SCORING_MODE=binary
```

CogVLM uses `COGVLM_CONDA_ENV_PATH`, defaulting to:

```bash
/projects/m000102/envs/levante-cogvlm
```

SmolVLM uses `SMOLVLM_CONDA_ENV_PATH`. If unset, the script tries to find an env with `num2words`, including:

```bash
/projects/m000102/envs/$USER-levante-py311
/projects/m000102/envs/david81-levante-py311
```

Override explicitly if needed:

```bash
SMOLVLM_CONDA_ENV_PATH=/projects/m000102/envs/david81-levante-py311 \
bash scripts/slurm/submit_forced_binary_all_tasks.sh
```

## Outputs

Default outputs are written under:

```bash
results/forced_binary_all_tasks/<model>/
```

Each model directory should contain per-task CSVs plus `summary.csv`.

To regenerate the comparison plot and dashboard override CSV after copying results back locally:

```bash
python scripts/analysis/plot_forced_binary_vocab_improvement.py
```

This writes:

- `results/analysis/vocab_forced_binary_improvement.png`
- `results/analysis/vocab_forced_binary_improvement.csv`
- `results/analysis/forced_binary_task_overrides.csv`
