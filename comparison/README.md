# LEVANTE comparison (R)

Statistical comparison of model outputs to human response data using **IRT-derived ability bins** (not raw age bins). Model runs once per `item_uid`; human response proportions are pre-aggregated by `item_uid` and 1-logit ability bins from fitted IRT models.

## Dependencies

- **R** with: `tidyverse`, `philentropy`, `nloptr`, `reticulate` (for .npy), `jsonlite` (for asset index)
- Install: `install.packages(c("tidyverse", "philentropy", "nloptr", "reticulate", "jsonlite"))`

## Scripts

- **stats-helper.R** – Softmax, KL, beta optimization, RSA (adapted from DevBench).
- **compare_levante.R** – Reads `data/responses/<version>/responses_by_ability/<task>_proportions_by_ability.csv` (item_uid, ability_bin, image1..image4) and `results/<version>/<model>/<task>.npy` (one row per item_uid). Joins IRT item difficulties from `data/responses/<version>/irt_models/<task>_item_params.csv`. Writes **D_KL** (per item_uid × ability_bin) and **accuracy** (per item_uid, with difficulty) to separate CSVs.

## IRT model mapping

The file `src/levante_bench/config/irt_model_mapping.csv` maps each task to its IRT model `.rds` file in the Redivis model registry. Columns: `task_id, model_file`. The download script reads this to know which `.rds` to fetch. Add rows manually for new tasks.

## Usage

1. **Preprocess human data (once):** Run the R download script to fetch trials, download IRT models, extract item difficulties and ability scores, and write ability-binned human proportions:

   ```bash
   Rscript scripts/download_levante_data.R [--version 2026-02-22] \
     [--irt-dataset levante_metadata_scoring:e97h:v1_11] \
     [--irt-table model_registry:rqwv]
   ```

   This produces:
   - `data/responses/<version>/irt_models/<task>.rds` – downloaded IRT model
   - `data/responses/<version>/irt_models/<task>_item_params.csv` – item difficulties (item_uid, difficulty)
   - `data/responses/<version>/irt_models/<task>_ability_scores.csv` – person abilities (run_id, ability, se)
   - `data/responses/<version>/responses_by_ability/<task>_proportions.csv` – overall response proportions (item_uid, image1..image4)
   - `data/responses/<version>/responses_by_ability/<task>_proportions_by_ability.csv` – ability-binned response proportions (item_uid, ability_bin, image1..image4), with 1-logit bins

2. **Run evaluation (Python):** One row per item_uid:

   ```bash
   levante-bench run-eval --task trog --model clip_base --version 2026-02-22
   ```

3. **Run comparison:**

   ```bash
   levante-bench run-comparison --task trog --model clip_base --version 2026-02-22 [--output-dir results/comparison]
   ```

   Or directly: `Rscript comparison/compare_levante.R --task trog --model clip_base --version 2026-02-22 --project-root .`

   Outputs (disaggregated):
   - **D_KL:** `results/comparison/<task>_<model>_d_kl.csv` — columns: task, model, item_uid, ability_bin, D_KL.
   - **Accuracy:** `results/comparison/<task>_<model>_accuracy.csv` — columns: task, model, item_uid, correct (0/1), difficulty.

4. **Optional: run full validation + benchmark smoke checks before comparison**

   ```bash
   scripts/validate_all.sh
   ```

5. **Optional: validate R/Redivis dependencies and comparison flow**

   ```bash
   scripts/validate_r.sh --check-packages-only
   scripts/validate_r.sh --run-comparison-smoke --task trog --model clip_base --version 2026-03-24
   ```

6. **Optional: inspect benchmark/prompt run history and metric deltas**

   ```bash
   python3 scripts/list_benchmark_results.py --limit 20
   ```

## Debugging the comparison flow

1. **Use one version everywhere**
   Use the same `--version` for: R download (trials + IRT → human_by_ability), Python run-eval, and run-comparison.

2. **Responses-by-ability must exist**
   Run `Rscript scripts/download_levante_data.R [--version VERSION]` so `data/responses/<version>/responses_by_ability/<task>_proportions_by_ability.csv` exists. The download script joins trials with IRT ability scores (from `@scores`), bins by 1-logit ability width, and aggregates response proportions by item_uid and ability_bin.

3. **IRT model mapping must be populated**
   Ensure `src/levante_bench/config/irt_model_mapping.csv` has a row for each task you want to compare. Without it, IRT models won't be downloaded and ability binning falls back to an "all" aggregate.

4. **Run evaluation (Python)**
   `levante-bench run-eval --task <TASK> --model <MODEL> --version <VERSION>`. The loader deduplicates by **item_uid**, so the .npy has one row per item_uid.

5. **Run comparison (R)**
   `levante-bench run-comparison --task <TASK> --model <MODEL> --version <VERSION>`. Writes D_KL and accuracy CSVs to `--output-dir` (default: results/comparison/).

## Sanity-checking the comparison

- **Item_uid alignment**
  The loader deduplicates by **item_uid**, so the model runs once per item and the .npy has one row per item_uid. The comparison aligns by item_uid (order from trials = order in .npy).

- **Accuracy**
  One row per item_uid: correct = 1 if model argmax (after softmax with fitted beta) equals the correct option, else 0. The `difficulty` column comes from the IRT model's `d` parameter. For 4 options, chance = 0.25. A negative correlation between `difficulty` and `correct` indicates the model finds harder items harder (expected).

- **D_KL**
  One row per (item_uid, ability_bin): KL(human proportions || model softmax) for that ability bin and item. Beta is fitted once to minimize mean D_KL across all (item_uid, ability_bin) pairs. Use the disaggregated D_KL CSV for per-ability or per-item analysis.

- **Difficulty correlation**
  The comparison script reports `difficulty correlation` — the point-biserial correlation between `correct` (0/1) and `difficulty` (IRT `d` parameter). Negative values mean harder items are less likely to be answered correctly by the model.

- **Spot-check**
  Inspect a few item_uids: in the accuracy CSV check that correct matches your expectation; in the D_KL CSV compare D_KL across ability bins or items.

## Vocab Graphics Bundle

Tracked vocab quadrant graphics live under `local_data/vocab_graphics/`.

- Graphics directory: `local_data/vocab_graphics/images/`
- Placement manifest: `local_data/vocab_graphics/vocab-quadrants-manifest.csv`
- Summary stats: `local_data/vocab_graphics/vocab-quadrants-summary.json`

Regenerate with:

```bash
python3 scripts/build_vocab_quadrant_graphics.py \
  --corpus-csv data/assets/2026-03-24/corpus/vocab/vocab-item-bank.csv \
  --visual-dir data/assets/2026-03-24/visual/vocab \
  --out-dir local_data/vocab_graphics/images \
  --manifest-csv local_data/vocab_graphics/vocab-quadrants-manifest.csv \
  --summary-json local_data/vocab_graphics/vocab-quadrants-summary.json \
  --seed 11
```
