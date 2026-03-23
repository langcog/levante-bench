# LEVANTE comparison (R)

Statistical comparison of model outputs to human response data by **item_uid** and (for D_KL) **age_bin**. Model runs once per item_uid; human data is pre-aggregated by item_uid and 1-year age bins (5–6, 6–7, …, 12–13).

## Dependencies

- **R** with: `tidyverse`, `philentropy`, `nloptr`, `reticulate` (for .npy), `jsonlite` (for asset index)
- Install: `install.packages(c("tidyverse", "philentropy", "nloptr", "reticulate", "jsonlite"))`

## Scripts

- **stats-helper.R** – Softmax, KL, beta optimization, RSA (adapted from DevBench).
- **compare_levante.R** – Reads `data/raw/<version>/human_by_age/<task>_proportions_by_age.csv` (item_uid, age_bin, image1..image4) and `results/<version>/<model>/<task>.npy` (one row per item_uid). Writes **D_KL** (per item_uid × age_bin) and **accuracy** (per item_uid) to separate CSVs.

## Usage

1. **Preprocess human data (once):** Run the R download script so trials are joined with scores (age) and human_by_age aggregates are written:

   ```bash
   Rscript scripts/download_levante_data.R [--version 2026-02-22] [--scores-table scores:pgms]
   ```

   This writes `data/raw/<version>/human_by_age/<task>_proportions_by_age.csv` (item_uid, age_bin, image1..image4), with age 5–12.99 and 1-year bins.

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
   - **D_KL:** `results/comparison/<task>_<model>_d_kl.csv` — columns: task, model, item_uid, age_bin, D_KL.
   - **Accuracy:** `results/comparison/<task>_<model>_accuracy.csv` — columns: task, model, item_uid, correct (0/1).

## Debugging the comparison flow

1. **Use one version everywhere**  
   Use the same `--version` for: R download (trials + scores → human_by_age), Python run-eval, and run-comparison.

2. **Human-by-age must exist**  
   Run `Rscript scripts/download_levante_data.R [--version VERSION]` so `data/raw/<version>/human_by_age/<task>_proportions_by_age.csv` exists. The download script joins trials with the scores table (run_id, age), filters 5 ≤ age ≤ 12.99, bins by year, and aggregates response proportions by item_uid and age_bin.

3. **Run evaluation (Python)**  
   `levante-bench run-eval --task <TASK> --model <MODEL> --version <VERSION>`. The loader deduplicates by **item_uid**, so the .npy has one row per item_uid. Check `results/<VERSION>/<model>/<task>.npy`.

4. **Run comparison (R)**  
   `levante-bench run-comparison --task <TASK> --model <MODEL> --version <VERSION>`. Writes D_KL and accuracy CSVs to `--output-dir` (default: results/comparison/). If R fails: ensure human_by_age files exist; .npy row count = number of unique item_uids in trials for that task.

## Sanity-checking the comparison

- **Item_uid alignment**  
  The loader deduplicates by **item_uid**, so the model runs once per item and the .npy has one row per item_uid. The comparison aligns by item_uid (order from trials = order in .npy).

- **Accuracy**  
  One row per item_uid: correct = 1 if model argmax (after softmax with fitted β) equals the correct option from the asset index (answer), else 0. Overall accuracy = mean(correct). For 4 options, chance = 0.25.

- **D_KL**  
  One row per (item_uid, age_bin): KL(human proportions ‖ model softmax) for that age bin and item. β is fitted once to minimize mean D_KL across all (item_uid, age_bin) pairs. Use the disaggregated D_KL CSV for per-age or per-item analysis.

- **Spot-check**  
  Inspect a few item_uids: in the accuracy CSV check that correct matches your expectation; in the D_KL CSV compare D_KL across age bins or items.
