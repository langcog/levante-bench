# Model run notes 4/30
Gemma4 31b timed out after 4 hours
  Nothing in the output file, so maybe retry
Molmo2 8b & 7b -- code issues
Tinyllava 2.4b -- FIXED! -- was missing stuff


# Slurm Launchers (Marlowe)

This folder contains Slurm wrappers for running LEVANTE experiments on Marlowe.

## Job status

Use the helper below for the default accounting view. It omits `Partition`
because Marlowe LEVANTE jobs use `preempt`, puts left-justified `JobName`
first, and shows `Reason` immediately after `State`.

```bash
bash scripts/slurm/sacct_recent.sh
```

Useful overrides:

```bash
START_TIME=2026-04-30 bash scripts/slurm/sacct_recent.sh
bash scripts/slurm/sacct_recent.sh --state=FAILED,TIMEOUT,CANCELLED
```

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
bash scripts/slurm/submit_all_local_models.sh internvl35:8B qwen35:4B tinyllava:3.1B
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

## Updated paper model list

Use this to submit the 23-model paper list on Marlowe. By default it skips
model-size labels that already have a `summary.csv` locally or in
`gs://levante-bench/results/<version>/<model-size>/baseline/`.

```bash
bash scripts/slurm/submit_paper_model_list.sh
```

Useful overrides:

```bash
DRY_RUN=1 bash scripts/slurm/submit_paper_model_list.sh
FORCE=1 bash scripts/slurm/submit_paper_model_list.sh
ONLY="internvl35:14B qwen35:27B molmo2:O-7B tinyllava:3.1B" bash scripts/slurm/submit_paper_model_list.sh
TIME=02:00:00 ONLY="molmo2:4B molmo2:O-7B molmo2:8B" bash scripts/slurm/submit_paper_model_list.sh
```

`TIME` is a global walltime override for every submitted target. If unset, the
launcher uses size-specific defaults so smaller models do not request the old
six-hour catch-all walltime.

Each job runs through `run_paper_model_baseline.sbatch`, writes deterministic
results to `results/<version>/<model-size>/`, and copies the CSV/JSON outputs
into `results/<version>/<model-size>/baseline/` for dashboard ingestion.

## Deterministic hosted-model launcher

Use this to run each hosted model once (no randomized option order) and write
to repo-local `results/<model_name>/...`:

```bash
bash scripts/slurm/submit_all_hosted_models_once.sh
```

Optional overrides:

```bash
VERSION=v1_new_parser bash scripts/slurm/submit_all_hosted_models_once.sh
```

## 100-run resampling launcher (10 x 10)

Use this to submit ten launches of ten true-random runs each, then stitch to a
single `0001..0100` sequence.

Default model is `qwen35-4B` (stronger than `smolvlm2-256M` while still
reasonable for repeated runs).

Submit:

```bash
bash scripts/slurm/submit_resampling_100.sh
```

Override model/version:

```bash
MODEL_NAME=internvl35 MODEL_SIZE=8B VERSION=v1_new_parser \
bash scripts/slurm/submit_resampling_100.sh
```

After all jobs finish, stitch chunked runs:

```bash
python scripts/analysis/stitch_resampling_runs.py \
  --source-root /projects/m000102/code/levante-bench/results/resampling/qwen35-4B/v1/qwen35-4B \
  --output-root /projects/m000102/code/levante-bench/results/resampling/qwen35-4B/v1/qwen35-4B_r0100
```

Resume preempted partial runs in place:

```bash
RUN_ROOT=/projects/m000102/code/levante-bench/results/resampling/qwen35-4B/v1/qwen35-4B \
BATCH_SIZE=2 MAX_NEW_TOKENS=1024 USE_JSON_FORMAT=true \
sbatch scripts/slurm/run_resume_resampling_partials.sbatch
```

To resume chunks in parallel, submit one resume job per `chunk_01..chunk_10`:

```bash
BATCH_SIZE=2 MAX_NEW_TOKENS=1024 USE_JSON_FORMAT=true \
bash scripts/slurm/submit_resume_resampling_chunks.sh
```

This launches separate jobs with `RUN_ROOT` set to each chunk directory, so each
job completes partial runs only within its own chunk.
By default it expects chunk folders under
`/projects/m000102/code/levante-bench/results/resampling/qwen35-4B/chunk_01`.
Set `CHUNKS_ROOT=/path/to/root-with-chunk-dirs` for another layout.

The resume job reads each partial run's `metadata.json`, reuses
`cache/responses.json`, skips task CSVs that already exist, evaluates missing
tasks with the original true-random run seed, and writes `summary.csv` when the
run becomes complete. Preview without loading a model:

```bash
python scripts/slurm/resume_resampling_partials.py \
  --run-root /projects/m000102/code/levante-bench/results/resampling/qwen35-4B/v1/qwen35-4B \
  --dry-run
```

## Large-model 40-run resampling launcher (40 x 1)

Use this workflow for large models where each Slurm job should perform only one
true-random run. It submits one job per model and chunk, defaults to
`TOTAL_RUNS=40`, limits each launcher invocation to `MAX_SUBMISSIONS=10`, and
writes chunked outputs to:

```text
/projects/m000102/code/levante-bench/results/resampling/<model-size>/v1/<model-size>/chunk_XX/0001/...
```

Preview the generated `sbatch` commands without submitting:

```bash
DRY_RUN=1 ONLY="gemma4:31B-it" \
bash scripts/slurm/submit_large_model_resampling_40.sh
```

Submit the next ten missing chunks for one large model:

```bash
ONLY="gemma4:31B-it" bash scripts/slurm/submit_large_model_resampling_40.sh
```

Submit several large models:

```bash
ONLY="gemma4:31B-it qwen35:27B internvl35:38B molmo2:8B" \
bash scripts/slurm/submit_large_model_resampling_40.sh
```

Useful overrides:

```bash
TOTAL_RUNS=40 MAX_SUBMISSIONS=10 TIME=08:00:00 MEM=180G GPUS=1 \
ONLY="qwen35:27B" bash scripts/slurm/submit_large_model_resampling_40.sh
```

Rerun the launcher safely after the active batch finishes to submit the next
missing chunks. Completed chunks with a `summary.csv` are skipped. Chunks with a
partial run folder are also skipped and should be resumed with the resume
launcher below. Set `MAX_SUBMISSIONS=0` only when you want to remove the
per-invocation cap.

Resume partial chunks in parallel:

```bash
ONLY="gemma4:31B-it" \
BATCH_SIZE=1 MAX_NEW_TOKENS=1024 USE_JSON_FORMAT=true \
bash scripts/slurm/submit_resume_large_model_resampling_40.sh
```

The resume launcher scans each `chunk_01..chunk_40` directory and submits a job
only when a run folder has `metadata.json` but is missing `summary.csv`. It
passes through `BATCH_SIZE`, `MAX_NEW_TOKENS`, `USE_JSON_FORMAT`, `TASKS_CSV`,
and `EXTRA_ARGS` so resume jobs can match the original model settings.

After all chunks complete, stitch them into a clean sequential `0001..0040`
folder:

```bash
python scripts/analysis/stitch_resampling_runs.py \
  --source-root results/resampling/gemma4-31B-it/v1/gemma4-31B-it \
  --output-root results/resampling/gemma4-31B-it/v1/gemma4-31B-it_r0040
```

## Historic VLM local launcher

Use this to submit one job each for the historic local VLM targets:
`llava15_13b`, `cogvlm`, and `openflamingo9b`.

```bash
bash scripts/slurm/submit_historic_vlms_local.sh
```

Preview generated `sbatch` commands without submitting:

```bash
DRY_RUN=1 bash scripts/slurm/submit_historic_vlms_local.sh
```

Submit only selected targets:

```bash
bash scripts/slurm/submit_historic_vlms_local.sh llava15_13b cogvlm
```

Useful overrides:

```bash
VERSION=v1_additional_images NUM_RUNS=1 \
OUTPUT_ROOT=/projects/m000102/code/levante-bench/results/v1_additional_models \
bash scripts/slurm/submit_historic_vlms_local.sh
```

The launcher uses `run_historic_vlm_local.sbatch` and per-model defaults for
walltime and memory, and follows the same conda pattern as other Marlowe
wrappers via `CONDA_ENV_PATH` (default: `/projects/m000102/envs/levante-bench-py311`).
Override via `CONDA_ENV_PATH=/projects/m000102/envs/your-env`.

After jobs finish, preview bucket sync (dry-run by default):

```bash
bash scripts/slurm/sync_historic_vlms_to_bucket.sh
```

Run real upload:

```bash
DRY_RUN=0 bash scripts/slurm/sync_historic_vlms_to_bucket.sh
```

If a remote machine has a partial checkout and is missing the local historic
adapter/config files, bootstrap them with a versioned script:

```bash
bash scripts/slurm/bootstrap_historic_local_adapters.sh
```

Use `FORCE=1` to overwrite existing files with repo templates.

## CogVLM from shared scratch

Use this workflow when CogVLM downloads are unstable or when runtime should avoid
network lookups entirely.

Prefetch required Hugging Face repos to shared scratch:

```bash
bash scripts/slurm/prefetch_cogvlm_to_scratch.sh
```

Submit full benchmark (all six tasks) using local snapshots:

```bash
bash scripts/slurm/submit_cogvlm_from_scratch.sh
```

Submit a smoke run (`vocab` only):

```bash
SMOKE=1 bash scripts/slurm/submit_cogvlm_from_scratch.sh
```

Run prefetch and submit in one command:

```bash
PREFETCH=1 bash scripts/slurm/submit_cogvlm_from_scratch.sh
```

By default these scripts use:
- env: `/projects/m000102/envs/levante-cogvlm`
- cache root: `/scratch/m000102/$USER/levante-hf`

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

## Hugging Face cache location

All Slurm wrappers that run through `scripts/slurm/_levante_common.sh` now
default model/tokenizer caches to shared scratch:

```text
/scratch/m000102/$USER/levante-hf/
  hf-home/
  hub/
  transformers/
```

Override with environment variables when needed:
`SCRATCH_ROOT`, `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`.
