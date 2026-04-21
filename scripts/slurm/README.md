# Slurm Launchers (Marlowe)

This folder contains Slurm wrappers for running LEVANTE experiments on Marlowe.

## Bulk local-model launcher

Use this to submit one job per local model with a consistent Marlowe setup:

```bash
bash scripts/slurm/submit_all_local_models.sh
```

Optional overrides:

```bash
VERSION=v1_new_parser NUM_RUNS=5 bash scripts/slurm/submit_all_local_models.sh
```

Submit only selected models:

```bash
bash scripts/slurm/submit_all_local_models.sh internvl35 qwen3vl_30b gemma4
```

## Deterministic single-run launcher

Use this to run each local model once (no randomized option order) and write
to repo-local `results/<model_name>/...`:

```bash
bash scripts/slurm/submit_all_local_models_once.sh
```

Optional overrides:

```bash
VERSION=v1_new_parser bash scripts/slurm/submit_all_local_models_once.sh
```

## What it runs

- Task set is fixed to all six benchmark tasks:
  - `egma-math`
  - `matrix-reasoning`
  - `mental-rotation`
  - `theory-of-mind`
  - `trog`
  - `vocab`
- Jobs are submitted through:
  - `scripts/slurm/run_local_model_experiment.sbatch`

## Output layout

Each model writes to a model-specific root, then job-specific folder:

```text
/projects/m000102/outputs/results/<model_name>/job_<SLURM_JOB_ID>/v1/<model-size-or-name>/...
```

Deterministic single-run launcher writes to:

```text
/projects/m000102/code/levante-bench/results/<model_name>/v1/<model-size-or-name>/...
```

Typical multirun path (true-random):

```text
/projects/m000102/outputs/results/internvl35/job_285999/v1/internvl35-8B/job285999-proc0/0001/summary.csv
```

## Logs

```text
/projects/m000102/outputs/slurm/<job-name>-<job-id>.out
/projects/m000102/outputs/slurm/<job-name>-<job-id>.err
```
