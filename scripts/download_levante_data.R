#!/usr/bin/env Rscript
# Download LEVANTE trials from Redivis and write to data/raw/<version>/.
# Joins scores (age) on run_id, filters 5 <= age <= 12.99, bins age by year,
# and writes per-task human response proportions by item_uid and age_bin.
# Usage: Rscript download_levante_data.R [--dataset DATASET] [--table TABLE] [--scores-table TABLE] [--version VERSION]
#   or set env: LEVANTE_DATASET, LEVANTE_TABLE, LEVANTE_SCORES_TABLE, LEVANTE_VERSION
# Default dataset: levante_data_pilots:68kn:v2_0, table: trials:ztnm, scores-table: scores:pgms

suppressPackageStartupMessages({
  library(tidyverse)
})

# Parse args: --dataset x --table y --version z
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NA_character_) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) return(Sys.getenv(paste0("LEVANTE_", toupper(name)), default))
  if (length(args) < i + 1L) return(default)
  args[i + 1L]
}

dataset_id   <- get_arg("dataset", "levante_data_pilots:68kn:v2_0")
table_name   <- get_arg("table", "trials:ztnm")
scores_table <- get_arg("scores-table", "scores:pgms")
version      <- get_arg("version", NA_character_)

if (is.na(version) || nchar(version) == 0L) {
  # Derive version from dataset id (e.g. v2_0 -> v2_0) or use "current"
  version <- if (grepl(":[a-z0-9_]+$", dataset_id)) {
    sub("^.*:", "", dataset_id)
  } else {
    "current"
  }
}

# Project root: directory containing data/ and scripts/
initial_options <- commandArgs(trailingOnly = FALSE)
file_arg <- initial_options[grepl("^--file=", initial_options)]
script_path <- if (length(file_arg) > 0L) sub("^--file=", "", file_arg[1L]) else "."
script_dir <- normalizePath(dirname(script_path), mustWork = FALSE)
if (is.na(script_dir) || nchar(script_dir) == 0L) script_dir <- "."
project_root <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
data_raw <- file.path(project_root, "data", "raw", version)
dir.create(data_raw, recursive = TRUE, showWarnings = FALSE)

# Redivis: user -> dataset -> table -> to_tibble()
# Requires redivis package and auth (see docs/releases.md)
if (!requireNamespace("redivis", quietly = TRUE)) {
  stop("Install the redivis R package. See https://docs.redivis.com/")
}
library(redivis)

user     <- redivis$user("levante")
dataset  <- user$dataset(dataset_id)
table    <- dataset$table(table_name)
d        <- table$to_tibble()

# Key columns to keep (align with docs/data_schema.md)
key_cols <- c(
  "redivis_source", "site", "dataset", "task_id", "user_id", "run_id",
  "trial_id", "trial_number", "item_uid", "item_task", "item_group", "item",
  "correct", "original_correct", "rt", "rt_numeric", "response", "response_type",
  "item_original", "answer", "distractors", "chance", "difficulty",
  "theta_estimate", "theta_se", "timestamp"
)
present <- intersect(key_cols, names(d))
trials  <- d %>% select(any_of(present))

# Load scores (run_id, age) and join
scores_tbl <- dataset$table(scores_table)
scores_df  <- scores_tbl$to_tibble()
if (!"run_id" %in% names(scores_df)) stop("Scores table must have run_id")
if (!"age" %in% names(scores_df)) stop("Scores table must have age")
scores_df <- scores_df %>% select(run_id, age) %>% distinct(run_id, .keep_all = TRUE)
trials <- trials %>% left_join(scores_df, by = "run_id")

# Write global trials CSV (with age if present)
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

# Human response proportions by item_uid and age_bin (5 <= age <= 12.99, 1-year bins)
# Exclude age < 5.0 and age > 12.99; age_bin e.g. "5-6", "6-7", ..., "12-13"
human_by_age_dir <- file.path(data_raw, "human_by_age")
dir.create(human_by_age_dir, recursive = TRUE, showWarnings = FALSE)
n_opts <- 4L

for (tid in tasks) {
  if (is.na(tid)) next
  safe_name <- gsub("[^a-zA-Z0-9_-]", "_", tid)
  task_trials <- trials %>%
    filter(task_id == !!tid, !is.na(response), response != "")
  if (!"age" %in% names(task_trials) || all(is.na(task_trials$age))) {
    message("  ", tid, ": no age column or all NA, skipping human_by_age")
    next
  }
  # Age-binned aggregates (5--12.99 only)
  task_trials_binned <- task_trials %>%
    filter(age >= 5, age <= 12.99) %>%
    mutate(age_bin = paste0(floor(age), "-", floor(age) + 1L))
  resp_levels <- sort(unique(task_trials$response))
  if (length(resp_levels) > n_opts) resp_levels <- resp_levels[seq_len(n_opts)]
  agg_binned <- if (nrow(task_trials_binned) > 0L) {
    task_trials_binned %>%
      mutate(option = match(response, resp_levels)) %>%
      filter(!is.na(option)) %>%
      group_by(item_uid, age_bin) %>%
      count(option, name = "n") %>%
      mutate(prop = n / sum(n, na.rm = TRUE)) %>%
      ungroup() %>%
      select(item_uid, age_bin, option, prop) %>%
      tidyr::pivot_wider(names_from = option, values_from = prop, names_prefix = "image") %>%
      mutate(across(starts_with("image"), ~ replace_na(., 0)))
  } else {
    tibble(item_uid = character(), age_bin = character())
  }
  # "all" age_bin: aggregate over all trials with response (so every item_uid has at least one row)
  task_trials_all <- task_trials %>% mutate(age_bin = "all")
  agg_all <- task_trials_all %>%
    mutate(option = match(response, resp_levels)) %>%
    filter(!is.na(option)) %>%
    group_by(item_uid, age_bin) %>%
    count(option, name = "n") %>%
    mutate(prop = n / sum(n, na.rm = TRUE)) %>%
    ungroup() %>%
    select(item_uid, age_bin, option, prop) %>%
    tidyr::pivot_wider(names_from = option, values_from = prop, names_prefix = "image") %>%
    mutate(across(starts_with("image"), ~ replace_na(., 0)))
  agg <- bind_rows(agg_binned, agg_all)
  if (nrow(agg) == 0L) {
    message("  ", tid, ": no rows, skipping human_by_age")
    next
  }
  for (j in seq_len(n_opts)) {
    col <- paste0("image", j)
    if (!col %in% names(agg)) agg[[col]] <- 0
  }
  agg <- agg %>% select(item_uid, age_bin, paste0("image", seq_len(n_opts)))
  n_uids_binned <- n_distinct(agg_binned$item_uid)
  n_uids_all <- n_distinct(agg_all$item_uid)
  if (nrow(agg_binned) > 0L && n_uids_binned <= 2L)
    message("  ", tid, ": only ", n_uids_binned, " item_uids in age bins 5--12.99 (check scores table run_id match)")
  out_path <- file.path(human_by_age_dir, paste0(safe_name, "_proportions_by_age.csv"))
  readr::write_csv(agg, out_path)
  message("Wrote ", out_path, " (", nrow(agg), " rows; ", n_uids_all, " item_uids with age_bin=all)")
}

message("Data version: ", version, " at ", data_raw)
