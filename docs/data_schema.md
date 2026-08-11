# Data schema

This document describes the canonical schema for LEVANTE trials data, human response aggregates, and the mapping from **item_uid** to assets (corpus and images).

## Redivis trials table

Trials data are obtained from Redivis via the **redivis** R package (not rlevante). Example R code:

```r
user <- redivis$user("levante")
dataset <- user$dataset("levante_data_latest:e9pf:v1_2")
table <- dataset$table("trials")
d <- table$to_tibble()
```

The result is a tibble of trial-level data. The **dataset** identifier (e.g. `levante_data_latest:e9pf:v1_2`) and **table** name (e.g. `trials`) may change for new releases; the R script accepts these as arguments or from config.

### Trials table columns

| Column | Description |
|--------|-------------|
| redivis_source | Redivis source identifier |
| site | Site identifier |
| dataset | Dataset identifier |
| task_id | Task identifier |
| user_id | User/participant identifier |
| run_id | Run (session) identifier |
| trial_id | Trial identifier |
| trial_number | Trial index within run |
| **item_uid** | **Key for mapping to assets** – identifies the task and the specific experimental item (LEVANTE "corpus" item) |
| item_task | Task label for item |
| item_group | Item group |
| item | Item label |
| correct | Whether response was correct |
| original_correct | Original correctness (if applicable) |
| rt | Response time |
| rt_numeric | Response time (numeric) |
| response | Response given |
| response_type | Type of response |
| item_original | Original item (if adapted) |
| answer | Correct answer |
| distractors | Distractor options (if applicable) |
| chance | Chance level |
| difficulty | IRT item `d` parameter if available; in current exports, higher values are empirically easier |
| theta_estimate | IRT ability estimate (if available) |
| theta_se | Standard error of theta (if available) |
| timestamp | Timestamp |

**item_uid** is the key column for joining trials to asset details (images, text) once assets are downloaded and indexed from the LEVANTE GCP bucket (see [releases.md](releases.md) and the asset download script).

## item_uid → corpus → assets

- **Trials** (from Redivis) include **item_uid** for each row.
- The LEVANTE **corpus** (per-task CSV files in the public bucket under `/corpus/<internal_name>/`) describe each experimental item and include **item_uid** and columns that specify prompts, answers, response alternatives, and (depending on task) image paths or keys.
- **Images** live in the bucket under `/visual/<internal_name>/...`. The exact mapping from corpus rows (e.g. `answer`, `response_alternatives`) to image object keys is task-specific and is defined in the task name mapping and, where needed, in the download script or config.
- After running `scripts/download_levante_assets.py`, the benchmark writes an **item_uid → local paths** index (e.g. CSV or JSON) under `data/assets/<version>/`. Loaders join trials by **item_uid** to resolve image and text paths for evaluation.

## Human response aggregates

For comparison with model outputs, human data may be aggregated by task and (optionally) age bin: e.g. trial (or item_uid), option index or label, proportion or count, age_bin. The R script and R comparison scripts consume trials and, where needed, produce or use such aggregates (e.g. per-task CSVs in `data/responses/<version>/responses_by_ability/`). Exact format follows DevBench-style layouts (e.g. trial, image1..imageN proportions, age_bin) as needed by the comparison scripts.
