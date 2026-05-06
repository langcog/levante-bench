# Croissant Metadata Guide

This repository now includes a validated Croissant metadata scaffold for LEVANTE v1:

- `datasets/v1/levante_v1.croissant.json`
- canonical schema/planning manifests in:
  - `datasets/v1/dataset.yaml`
  - `datasets/vocab/dataset.yaml`

## What has been implemented

- Croissant 1.1-compatible JSON-LD context and metadata structure.
- Multiple related RecordSets:
  - `assets_items` keyed by `item_uid`
  - `responses_trials` keyed by `trial_id`, linked to assets via `item_uid`
  - `responses_by_ability` (optional aggregate-style RecordSet)
- Distribution entries with `contentUrl`, `encodingFormat`, and `sha256`.
- RAI/provenance-style metadata fields for collection context and limitations.
- NeurIPS-targeted RAI fields included in JSON-LD: `rai:dataBiases`,
  `rai:dataSocialImpact`, and `rai:hasSyntheticData`.
- Validation workflow using `mlcroissant` (currently passes with 0 errors / 0 warnings).

## Licensing split (code vs data)

- Repository code/docs are licensed under MIT (see `LICENSE`).
- LEVANTE benchmark assets/data referenced by Croissant are noncommercial-use
  only unless a file-specific notice states otherwise.
- `datasets/v1/levante_v1.croissant.json` sets dataset license to
  `https://creativecommons.org/licenses/by-nc/4.0/` for the distributed data
  artifacts.

## How to keep Croissant up to date

Croissant needs updates only when files referenced in `distribution` change.
Adding model outputs under `results/` does not require Croissant edits.

### When you must update

- You regenerate any referenced assets/responses manifests.
- You change distribution URLs.
- You change schema/column definitions used by RecordSets.

### Maintenance workflow

1. Refresh source data/manifests (assets + responses) as needed.
2. Recompute distribution checksums:

```bash
PYTHONPATH=src .venv/bin/python scripts/data_prep/update_croissant_checksums.py \
  --croissant-json datasets/v1/levante_v1.croissant.json
```

3. Validate metadata:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import mlcroissant as mlc
ds = mlc.Dataset("datasets/v1/levante_v1.croissant.json")
print("errors", len(ds.metadata.ctx.issues.errors))
print("warnings", len(ds.metadata.ctx.issues.warnings))
PY
```

4. Commit any Croissant changes together with the data refresh.

## Check-only mode (CI-friendly)

Use `--check` to verify whether checksum updates are required without writing:

```bash
PYTHONPATH=src .venv/bin/python scripts/data_prep/update_croissant_checksums.py \
  --croissant-json datasets/v1/levante_v1.croissant.json \
  --check
```

- Exit code `0`: no checksum updates needed.
- Exit code `1`: checksum updates are needed.
- Exit code `2`: one or more `contentUrl` entries could not be resolved locally.

## Notes

- Asset distributions currently point to `levante-bench` bucket URLs.
- Response distributions currently use raw GitHub URLs and are resolved locally for checksum refresh.
- If distribution paths change, update both `contentUrl` and local path resolution logic in:
  - `scripts/data_prep/update_croissant_checksums.py`

## Bucket publishing for assets manifests

Assets are published from the LEVANTE bucket path used by the asset downloader
(`https://storage.googleapis.com/levante-bench/corpus_data`).

To ensure Croissant asset `contentUrl` entries stay resolvable, publish:

- `data/assets/manifest.csv` -> `gs://levante-bench/corpus_data/v1/manifest.csv`
- `data/assets/v1/manifests/*` -> `gs://levante-bench/corpus_data/v1/manifests/`
- checksums sidecar generated as:
  - `data/assets/v1/manifests/CROISSANT-SHA256SUMS.txt`

Use:

```bash
PYTHONPATH=src .venv/bin/python scripts/data_prep/publish_assets_manifests_to_bucket.py \
  --version v1 \
  --bucket-url https://storage.googleapis.com/levante-bench/corpus_data
```

Dry-run:

```bash
PYTHONPATH=src .venv/bin/python scripts/data_prep/publish_assets_manifests_to_bucket.py \
  --version v1 \
  --bucket-url https://storage.googleapis.com/levante-bench/corpus_data \
  --dry-run
```

## Portals and automated checkers (e.g. NeurIPS)

**The manifest files are on the bucket** under these public HTTPS URLs (also listed as `distribution[].contentUrl` in the Croissant file):

| Prefix on `gs://levante-bench/` | Example object |
|---------------------------------|----------------|
| `corpus_data/v1/` | `manifest.csv`, `manifests/v1_eval_all.parquet` |
| `responses_manifests/v1/` | `trials.csv`, `manifests/trials_all.parquet`, `manifests/responses_by_ability_all.parquet` |

They are **not** at the bucket root; tools that only **list** `gs://levante-bench/` without reading Croissant may appear to find “no manifests.”

**Give the portal the machine-readable Croissant URL**, not the GitHub **blob** (HTML) page:

- Use: `https://raw.githubusercontent.com/langcog/levante-bench/main/datasets/v1/levante_v1.croissant.json`
- Avoid: `https://github.com/langcog/levante-bench/blob/main/datasets/v1/levante_v1.croissant.json` (returns HTML; fetchers that resolve `@id` will not see JSON).

The dataset `@id` / `citeAs` in repo point at the **raw** URL so automated validators can download JSON-LD and follow `distribution` links.

## Responses note

Response source data is downloaded from Redivis (`download_levante_data.R`).
Downloader output now defaults to `data/responses/v1/` and includes a local
`SHA256SUMS.txt` sidecar for reproducibility checks.
Croissant response distributions are currently bucket-hosted under:

- `https://storage.googleapis.com/levante-bench/responses_manifests/v1/...`

If response manifests are refreshed, rerun:

1. `update_croissant_checksums.py`
2. `mlcroissant` validation
