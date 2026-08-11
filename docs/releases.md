# Releases and data ingestion

This document describes how to obtain LEVANTE trials data and assets, and how versioning works.

## Redivis (trials data)

Trials data are accessed via the **redivis** R package. Install and authenticate according to the [Redivis documentation](https://docs.redivis.com/).

### R code pattern

```r
user <- redivis$user("levante")
dataset <- user$dataset("levante_data_latest:e9pf:v1_2")   # pin v1_2+; dataset id may change
table <- dataset$table("trials:bxf8")                       # table name may change
d <- table$to_tibble()
```

The R script `scripts/data_prep/download_levante_data.R` accepts **dataset** (default `levante_data_latest:e9pf:v1_2`), **table** (default `trials:bxf8`), and optional **version** (default: local `v2` for latest / pilots v2). It writes trials (including key columns such as `task_id`, `trial_id`, `item_uid`, `response`, `correct`) to `data/responses/<version>/` and emits `data/responses/<version>/SHA256SUMS.txt` plus `SOURCE.json` for reproducibility.

### New releases

To use a new Redivis release, run the R script with the new dataset/table identifiers (or version id). Output is written to `data/responses/<version>/`. The Python benchmark and R comparison scripts use the same version string (e.g. `--version <version>`) to point at the correct data and assets.

## LEVANTE assets (public bucket)

Assets (corpus CSVs and images) are in a **public** GCP bucket. No authentication is required; the download script uses standard HTTP.

- **Base URL:** `https://storage.googleapis.com/levante-assets-prod` (configurable in `levante_bench.config.defaults` or env `LEVANTE_ASSETS_BUCKET_URL`).
- **Corpus (item mapping):** `{base}/corpus/{internal_name}/{corpus_file}` – e.g. `corpus/egma-math/math-item-bank.csv`. The mapping from benchmark task name to internal name and corpus filename is in `config/task_name_mapping.csv`.
- **Images:** `{base}/visual/{internal_name}/...` – per-task subdirectories.

Run `scripts/download_levante_assets.py` with optional `--version YYYY-MM-DD` (default: today). Assets are written to `data/assets/<version>/` and an item_uid → paths index is built. Production assets change over time, so versioning by **download date** is used.

## Versioning summary

| Data | Version key | Location |
|------|-------------|----------|
| Trials (Redivis) | local folder via `--version` (default **v2** for Redivis `v2_0`) | `data/responses/<version>/` |
| Assets (bucket) | Download date (YYYY-MM-DD) | `data/assets/<version>/` |

**Redivis release tag ≠ local folder.** `levante_data_latest:e9pf:v1_2` (unified all-sites; preferred) and the older pilots `v2_0` export both write to **`data/responses/v2/`**. The frozen April tree stays at **`data/responses/v1/`**. The download script refuses to write current/latest pulls into local `v1/`.

Each download writes `data/responses/<version>/SOURCE.json` with dataset/table/IRT ids and timestamp.

When running evaluation or comparison, use a consistent version (or "latest") so that trials and assets align (e.g. same task set and item_uid space). Default analysis / persona builders use **responses v2**.
