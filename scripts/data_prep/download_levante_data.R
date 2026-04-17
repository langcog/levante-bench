#!/usr/bin/env Rscript
# Download LEVANTE trials and IRT models from Redivis, extract item difficulties
# and person ability scores, bin by ability (1-logit width), and write per-task
# human response proportions by item_uid and ability_bin.
# Usage: Rscript download_levante_data.R [--dataset DATASET] [--table TABLE] [--scores-table TABLE]
#        [--irt-dataset DATASET] [--irt-table TABLE] [--version VERSION]
#   or set env: LEVANTE_DATASET, LEVANTE_TABLE, LEVANTE_SCORES_TABLE, LEVANTE_VERSION

suppressPackageStartupMessages({
  library(tidyverse)
  library(here)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NA_character_) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) return(Sys.getenv(paste0("LEVANTE_", toupper(name)), default))
  if (length(args) < i + 1L) return(default)
  args[i + 1L]
}

dataset_id    <- get_arg("dataset", "levante_data_pilots:68kn:v2_0")
table_name    <- get_arg("table", "trials:ztnm")
scores_table  <- get_arg("scores-table", "scores:pgms")
irt_dataset   <- get_arg("irt-dataset", "levante_metadata_scoring:e97h:v1_11")
irt_table     <- get_arg("irt-table", "model_registry:rqwv")
version       <- get_arg("version", NA_character_)

if (is.na(version) || nchar(version) == 0L) {
  version <- format(Sys.Date(), "%Y-%m-%d")
}

data_raw <- here("data", "responses", version)
dir.create(data_raw, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("redivis", quietly = TRUE)) {
  stop("Install the redivis R package. See https://docs.redivis.com/")
}
options(expressions = 500000)
library(redivis)

# ── 1. Download trials ────────────────────────────────────────────────────────

user     <- redivis$user("levante")
dataset  <- user$dataset(dataset_id)
table    <- dataset$table(table_name)
d        <- table$to_tibble(max_results = 5000000)

key_cols <- c(
  "redivis_source", "site", "dataset", "task_id", "user_id", "run_id",
  "trial_id", "trial_number", "item_uid", "item_task", "item_group", "item",
  "correct", "original_correct", "rt", "rt_numeric", "response", "response_type",
  "item_original", "answer", "distractors", "chance", "difficulty",
  "theta_estimate", "theta_se", "timestamp"
)
present <- intersect(key_cols, names(d))
trials  <- d %>% select(any_of(present))

# Join age and site from scores table
scores_tbl <- dataset$table(scores_table)
scores_df  <- scores_tbl$to_tibble(max_results = 5000000)
scores_cols <- intersect(c("run_id", "age", "site"), names(scores_df))
if ("run_id" %in% scores_cols && length(scores_cols) > 1L) {
  scores_join <- scores_df %>% select(all_of(scores_cols)) %>% distinct(run_id, .keep_all = TRUE)
  # If trials already has these columns (possibly all-NA), drop them before joining
  overlap <- setdiff(intersect(names(trials), names(scores_join)), "run_id")
  if (length(overlap) > 0L) trials <- trials %>% select(-all_of(overlap))
  trials <- trials %>% left_join(scores_join, by = "run_id")
}

trials_path <- file.path(data_raw, "trials.csv")
readr::write_csv(trials, trials_path)
message("Wrote ", trials_path, " (", nrow(trials), " rows)")

# Per-task trials
tasks <- unique(trials$task_id)
tasks_dir <- file.path(data_raw, "tasks")
dir.create(tasks_dir, recursive = TRUE, showWarnings = FALSE)
for (tid in tasks) {
  if (is.na(tid)) next
  task_trials <- trials %>% filter(task_id == !!tid)
  safe_name <- gsub("[^a-zA-Z0-9_-]", "_", tid)
  readr::write_csv(task_trials, file.path(tasks_dir, paste0(safe_name, "_trials.csv")))
}
message("Wrote per-task trials to ", tasks_dir)

# ── 2. Download IRT models and extract parameters ─────────────────────────────

irt_mapping_path <- here("src", "levante_bench", "config", "irt_model_mapping.csv")
irt_mapping <- if (file.exists(irt_mapping_path)) {
  readr::read_csv(irt_mapping_path, show_col_types = FALSE) %>%
    filter(!is.na(task_id), !is.na(model_file), nzchar(task_id), nzchar(model_file))
} else {
  tibble(task_id = character(), model_file = character(), site = character())
}

if (nrow(irt_mapping) == 0L) {
  message("No IRT model mapping (or empty); skipping IRT download")
} else {
    irt_ds  <- user$dataset(irt_dataset)
    irt_tbl <- irt_ds$table(irt_table)
    registry <- irt_tbl$to_tibble(max_results = 500)
    message("Model registry: ", nrow(registry), " entries")

    irt_dir <- file.path(data_raw, "irt_models")
    dir.create(irt_dir, recursive = TRUE, showWarnings = FALSE)

    for (row_i in seq_len(nrow(irt_mapping))) {
      tid        <- irt_mapping$task_id[row_i]
      model_file <- trimws(irt_mapping$model_file[row_i])
      safe_name  <- gsub("[^a-zA-Z0-9_-]", "_", tid)

      reg_row <- registry %>% filter(file_name == model_file)
      if (nrow(reg_row) == 0L) {
        message("  ", tid, ": model_file '", model_file, "' not found in registry; skipping")
        next
      }
      fid <- reg_row$file_id[1]

      # Download .rds via stream
      rds_path <- file.path(irt_dir, paste0(safe_name, ".rds"))
      if (!file.exists(rds_path)) {
        con <- base::file(rds_path, "wb")
        irt_tbl$file(fid)$stream(callback = function(chunk) writeBin(chunk, con))
        close(con)
        message("  ", tid, ": downloaded ", rds_path)
      } else {
        message("  ", tid, ": ", rds_path, " already exists")
      }

      mod <- readRDS(rds_path)

      # Extract item difficulties (d parameter)
      model_vals <- attr(mod, "model_vals")
      group_names <- attr(mod, "group_names")
      ref_group <- group_names[1]
      d_params <- model_vals %>%
        filter(name == "d", group == ref_group) %>%
        mutate(item_uid = gsub("-1$", "", item)) %>%
        select(item_uid, difficulty = value)
      params_path <- file.path(irt_dir, paste0(safe_name, "_item_params.csv"))
      readr::write_csv(d_params, params_path)
      message("  ", tid, ": wrote ", params_path, " (", nrow(d_params), " items)")

      # Extract ability scores (run_id, ability, se)
      irt_scores <- attr(mod, "scores")
      if (!is.data.frame(irt_scores) || !"run_id" %in% names(irt_scores)) {
        message("  ", tid, ": no scores in IRT model; skipping ability binning")
        next
      }
      irt_scores <- irt_scores %>%
        select(run_id, ability, any_of("se")) %>%
        distinct(run_id, .keep_all = TRUE)
      scores_path <- file.path(irt_dir, paste0(safe_name, "_ability_scores.csv"))
      readr::write_csv(irt_scores, scores_path)
      message("  ", tid, ": wrote ", scores_path, " (", nrow(irt_scores), " runs, ability range ",
              round(min(irt_scores$ability, na.rm = TRUE), 2), " to ",
              round(max(irt_scores$ability, na.rm = TRUE), 2), ")")
    }
}

# ── 3. Response proportions by item_uid (and ability_bin) ──────────────────────

n_opts <- 4L
responses_dir <- file.path(data_raw, "responses_by_ability")
dir.create(responses_dir, recursive = TRUE, showWarnings = FALSE)

# Parse distractors from Python-dict-like string: {'0': 'coconut', '1': 'key', '2': 'clothesline'}
# Returns a character vector in key order (0, 1, 2, ...).
parse_distractors <- function(s) {
  if (is.null(s) || is.na(s) || !nzchar(trimws(as.character(s)))) return(character(0))
  s <- trimws(as.character(s))
  s <- gsub("^\\{|\\}$", "", s)
  pairs <- strsplit(s, ",(?=\\s*['\"])", perl = TRUE)[[1]]
  vals <- character(0)
  for (pair in pairs) {
    kv <- strsplit(pair, ":\\s*", perl = TRUE)[[1]]
    if (length(kv) >= 2L) {
      val <- trimws(paste(kv[-1], collapse = ":"))
      val <- gsub("^['\"]|['\"]$", "", val)
      vals <- c(vals, val)
    }
  }
  vals
}

# Build canonical option map per item_uid: image1 = answer, image2.. = distractors in order.
# Returns a tibble with columns: item_uid, response, option (1..n_opts).
build_option_map <- function(task_df, n_opts) {
  first_per_item <- task_df %>%
    filter(!is.na(item_uid), item_uid != "", !is.na(answer)) %>%
    distinct(item_uid, .keep_all = TRUE)

  map_rows <- list()
  for (i in seq_len(nrow(first_per_item))) {
    uid <- first_per_item$item_uid[i]
    ans <- as.character(first_per_item$answer[i])
    dist_str <- if ("distractors" %in% names(first_per_item)) first_per_item$distractors[i] else NA
    dists <- parse_distractors(dist_str)
    opts <- c(ans, dists)
    opts <- opts[seq_len(min(length(opts), n_opts))]
    map_rows[[i]] <- tibble(
      item_uid = uid,
      response = opts,
      option = seq_along(opts)
    )
  }
  bind_rows(map_rows)
}

# Aggregate response proportions using a pre-built option map.
aggregate_proportions <- function(df, bin_col, option_map, n_opts) {
  agg <- df %>%
    inner_join(option_map, by = c("item_uid", "response")) %>%
    group_by(item_uid, !!sym(bin_col)) %>%
    count(option, name = "n") %>%
    mutate(prop = n / sum(n, na.rm = TRUE)) %>%
    ungroup() %>%
    select(item_uid, !!sym(bin_col), option, prop) %>%
    tidyr::pivot_wider(names_from = option, values_from = prop, names_prefix = "image") %>%
    mutate(across(starts_with("image"), ~ replace_na(., 0)))
  for (j in seq_len(n_opts)) {
    col <- paste0("image", j)
    if (!col %in% names(agg)) agg[[col]] <- 0
  }
  agg %>% select(item_uid, !!sym(bin_col), paste0("image", seq_len(n_opts)))
}

for (tid in tasks) {
  if (is.na(tid)) next
  safe_name <- gsub("[^a-zA-Z0-9_-]", "_", tid)

  task_trials <- trials %>%
    filter(task_id == !!tid, !is.na(response), response != "")
  if (nrow(task_trials) == 0L) next

  # Filter by site if irt_model_mapping specifies one
  irt_row <- irt_mapping %>% filter(task_id == !!tid)
  if (nrow(irt_row) > 0L && "site" %in% names(irt_row)) {
    site_filter <- trimws(irt_row$site[1])
    if (!is.na(site_filter) && nzchar(site_filter) && site_filter != "all") {
      if ("site" %in% names(task_trials)) {
        n_before <- nrow(task_trials)
        task_trials <- task_trials %>% filter(site == site_filter)
        message("  ", tid, ": filtered to site '", site_filter, "' (",
                n_before, " → ", nrow(task_trials), " trials)")
        if (nrow(task_trials) == 0L) next
      } else {
        message("  ", tid, ": site filter '", site_filter, "' requested but no site column in trials")
      }
    }
  }

  # Build canonical option map: image1 = answer, image2..N = distractors in order
  option_map <- build_option_map(task_trials, n_opts)
  if (nrow(option_map) == 0L) {
    message("  ", tid, ": could not build option map (no answer/distractors); skipping")
    next
  }

  # Write option key legend: item_uid, image1, image2, ..., imageN (response labels)
  option_key <- option_map %>%
    tidyr::pivot_wider(names_from = option, values_from = response, names_prefix = "image") %>%
    select(item_uid, any_of(paste0("image", seq_len(n_opts))))
  key_path <- file.path(responses_dir, paste0(safe_name, "_option_key.csv"))
  readr::write_csv(option_key, key_path)
  message("Wrote ", key_path, " (", nrow(option_key), " item_uids; image1 = target)")

  # Try to join ability scores from IRT model
  ability_path <- file.path(data_raw, "irt_models", paste0(safe_name, "_ability_scores.csv"))
  has_ability <- file.exists(ability_path)
  if (has_ability) {
    irt_scores <- readr::read_csv(ability_path, show_col_types = FALSE)
    task_trials <- task_trials %>%
      left_join(irt_scores %>% select(run_id, ability), by = "run_id")
  }

  # Ability-binned aggregates (1-logit bins)
  if (has_ability && "ability" %in% names(task_trials) && !all(is.na(task_trials$ability))) {
    task_trials_binned <- task_trials %>%
      filter(!is.na(ability)) %>%
      mutate(ability_bin = paste0(floor(ability), "_", floor(ability) + 1L))
    if (nrow(task_trials_binned) > 0L) {
      agg_binned <- aggregate_proportions(task_trials_binned, "ability_bin", option_map, n_opts)
    } else {
      agg_binned <- tibble(item_uid = character(), ability_bin = character())
    }
    n_uids_binned <- n_distinct(agg_binned$item_uid)
  } else {
    agg_binned <- tibble(item_uid = character(), ability_bin = character())
    n_uids_binned <- 0L
  }

  # Overall proportions (all trials, no binning) → separate file
  task_trials_all <- task_trials %>% mutate(ability_bin = "all")
  agg_all <- aggregate_proportions(task_trials_all, "ability_bin", option_map, n_opts) %>%
    select(-ability_bin)
  n_uids_all <- n_distinct(agg_all$item_uid)

  # Write overall proportions
  overall_path <- file.path(responses_dir, paste0(safe_name, "_proportions.csv"))
  readr::write_csv(agg_all, overall_path)
  message("Wrote ", overall_path, " (", n_uids_all, " item_uids)")

  # Write ability-binned proportions (only rows with actual ability bins)
  if (nrow(agg_binned) > 0L) {
    binned_path <- file.path(responses_dir, paste0(safe_name, "_proportions_by_ability.csv"))
    readr::write_csv(agg_binned, binned_path)
    message("Wrote ", binned_path, " (", nrow(agg_binned), " rows; ",
            n_uids_binned, " item_uids)")
  } else {
    message("  ", tid, ": no ability-binned rows (no IRT scores matched)")
  }
}

message("Data version: ", version, " at ", data_raw)
