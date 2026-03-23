# LEVANTE comparison: human_by_age (item_uid × age_bin) vs model (one row per item_uid).
# D_KL per (item_uid, age_bin); accuracy per item_uid. Writes disaggregated CSVs.
# Usage: Rscript compare_levante.R --task TASK --model MODEL [--version VERSION] [--output-dir DIR] [--output-dkl CSV] [--output-accuracy CSV]
# Requires: tidyverse, philentropy, nloptr, reticulate, jsonlite (for asset index)

library(tidyverse)
library(reticulate)
library(jsonlite)

# Source stats-helper from comparison/
script_dir <- if (length(commandArgs(trailingOnly = FALSE)) > 0) {
  arg <- commandArgs(trailingOnly = FALSE)[grepl("^--file=", commandArgs(trailingOnly = FALSE))]
  if (length(arg) > 0) dirname(sub("^--file=", "", arg[1])) else "."
} else "."
source(file.path(script_dir, "stats-helper.R"), local = TRUE)

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NA_character_) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) return(Sys.getenv(paste0("LEVANTE_", toupper(gsub("-", "_", name))), default))
  if (length(args) < i + 1L) return(default)
  args[i + 1L]
}

version      <- get_arg("version", "current")
results_dir  <- get_arg("results-dir", "results")
task_id      <- get_arg("task", NA_character_)
model_id     <- get_arg("model", NA_character_)
output_dir   <- get_arg("output-dir", "results/comparison")
output_dkl   <- get_arg("output-dkl", NA_character_)
output_acc   <- get_arg("output-accuracy", NA_character_)
project_root <- get_arg("project-root", ".")

data_raw     <- file.path(project_root, "data", "raw", version)
data_assets  <- file.path(project_root, "data", "assets", version)
results_base <- file.path(project_root, results_dir, version)

safe_task  <- gsub("[^a-zA-Z0-9_-]", "_", task_id)
safe_model <- gsub("[^a-zA-Z0-9_-]", "_", model_id)

# Default output paths if not specified
if (is.na(output_dkl) || nchar(output_dkl) == 0L)
  output_dkl <- file.path(output_dir, paste0(safe_task, "_", safe_model, "_d_kl.csv"))
if (is.na(output_acc) || nchar(output_acc) == 0L)
  output_acc <- file.path(output_dir, paste0(safe_task, "_", safe_model, "_accuracy.csv"))

# Item_uid order: same as Python manifest (unique item_uid in order of first appearance in trials)
get_item_uid_order <- function(task_id, version, project_root) {
  tasks_dir  <- file.path(project_root, "data", "raw", version, "tasks")
  trials_path <- file.path(project_root, "data", "raw", version, "trials.csv")
  trials_file <- file.path(tasks_dir, paste0(safe_task, "_trials.csv"))
  if (file.exists(trials_file)) {
    d <- read_csv(trials_file, show_col_types = FALSE)
  } else if (file.exists(trials_path)) {
    d <- read_csv(trials_path, show_col_types = FALSE) %>% filter(task_id == !!task_id)
  } else {
    stop("No trials found at ", trials_path, " or ", trials_file)
  }
  unique(d[["item_uid"]])
}

# Load human proportions by item_uid and age_bin
get_human_by_age <- function(task_id, version, project_root) {
  human_dir <- file.path(project_root, "data", "raw", version, "human_by_age")
  path <- file.path(human_dir, paste0(safe_task, "_proportions_by_age.csv"))
  if (!file.exists(path)) stop("Human-by-age file not found: ", path, " (run download_levante_data.R first)")
  read_csv(path, show_col_types = FALSE)
}

# Load model .npy and attach item_uid (one row per item_uid)
load_model_by_item <- function(results_base, task_id, model_id, item_uid_order) {
  npy_path <- file.path(results_base, safe_model, paste0(safe_task, ".npy"))
  if (!file.exists(npy_path)) stop("Model .npy not found: ", npy_path)
  np <- import("numpy")
  m <- np$load(npy_path)
  if (length(dim(m)) == 3L) m <- m[,,1] - m[,,2]
  d <- as_tibble(as.matrix(m))
  n_opts <- ncol(d)
  names(d) <- paste0("image", seq_len(n_opts))
  d %>%
    mutate(row = row_number(), item_uid = item_uid_order[row]) %>%
    select(item_uid, starts_with("image"))
}

# Correct option (1..n) per item_uid from asset index (option order = image_paths or answer + response_alternatives)
get_correct_option_per_item <- function(version, project_root, item_uids) {
  index_path <- file.path(project_root, "data", "assets", version, "item_uid_index.json")
  if (!file.exists(index_path)) return(NULL)
  index <- jsonlite::read_json(index_path, simplifyVector = TRUE)
  out <- tibble(
    item_uid = character(length(item_uids)),
    correct_option = integer(length(item_uids))
  )
  for (i in seq_along(item_uids)) {
    uid <- item_uids[i]
    ent <- index[[uid]]
    if (is.null(ent)) { out$correct_option[i] <- NA_integer_; out$item_uid[i] <- uid; next }
    cr <- ent$corpus_row
    if (is.null(cr)) { out$correct_option[i] <- NA_integer_; out$item_uid[i] <- uid; next }
    answer <- cr$answer
    if (is.null(answer)) { out$correct_option[i] <- NA_integer_; out$item_uid[i] <- uid; next }
    paths <- ent$image_paths
    if (!is.null(paths) && length(paths) > 0L) {
      opts <- gsub("\\.[a-zA-Z0-9]+$", "", basename(paths))
    } else {
      alts <- cr$response_alternatives
      if (is.null(alts)) alts <- ""
      opts <- c(answer, strsplit(trimws(alts), "\\s*,\\s*")[[1]])
      opts <- opts[nzchar(opts)]
    }
    out$item_uid[i] <- uid
    out$correct_option[i] <- match(answer, opts)[1]
  }
  out
}

# Fallback: correct option from trials (answer + distractors) when index lacks item or returns NA
get_correct_option_from_trials <- function(task_id, version, project_root, item_uids) {
  tasks_dir <- file.path(project_root, "data", "raw", version, "tasks")
  trials_path <- file.path(project_root, "data", "raw", version, "trials.csv")
  trials_file <- file.path(tasks_dir, paste0(safe_task, "_trials.csv"))
  if (file.exists(trials_file)) {
    d <- read_csv(trials_file, show_col_types = FALSE)
  } else if (file.exists(trials_path)) {
    d <- read_csv(trials_path, show_col_types = FALSE) %>% filter(task_id == !!task_id)
  } else return(NULL)
  if (!"answer" %in% names(d)) return(NULL)
  alt_col <- if ("response_alternatives" %in% names(d)) "response_alternatives" else "distractors"
  if (!alt_col %in% names(d)) alt_col <- NULL
  first <- d %>%
    filter(!is.na(item_uid), item_uid != "", !is.na(answer)) %>%
    group_by(item_uid) %>%
    slice(1L) %>%
    ungroup()
  if (nrow(first) == 0L) return(NULL)
  parse_alternatives <- function(alt_str) {
    if (is.null(alt_str) || is.na(alt_str) || !nzchar(trimws(as.character(alt_str)))) return(character(0))
    s <- trimws(as.character(alt_str))
    if (grepl("['\"]", s)) {
      m <- gregexpr("['\"]([^'\"]+)['\"]", s)
      as.character(regmatches(s, m)[[1]]) %>% gsub("^['\"]|['\"]$", "", .)
    } else {
      strsplit(s, "\\s*,\\s*")[[1]]
    }
  }
  first %>%
    mutate(
      alts_vec = if (!is.null(alt_col)) map(!!sym(alt_col), parse_alternatives) else rep(list(character(0)), n()),
      opts = map2(as.character(answer), alts_vec, function(a, b) c(a, b)),
      correct_option = as.integer(map2_dbl(opts, as.character(answer), function(o, a) match(a, o)[1]))
    ) %>%
    select(item_uid, correct_option)
}

# Compare one task / one model: D_KL by (item_uid, age_bin), accuracy by item_uid
compare_one <- function(task_id, model_id, version, results_base, project_root) {
  item_uid_order <- get_item_uid_order(task_id, version, project_root)
  human <- get_human_by_age(task_id, version, project_root)
  model_wide <- load_model_by_item(results_base, task_id, model_id, item_uid_order)

  # Align: model has one row per item_uid; human has multiple rows per item_uid (one per age_bin)
  n_opts <- length(grep("^image[0-9]+$", names(model_wide), value = TRUE))
  img_cols <- paste0("image", seq_len(n_opts))

  # Fit beta: minimize mean KL over all (item_uid, age_bin) pairs (human proportions vs softmax(model logits))
  human_joined_opt <- human %>%
    inner_join(model_wide %>% select(item_uid, all_of(img_cols)), by = "item_uid", suffix = c("_h", "_m"))
  if (nrow(human_joined_opt) == 0L) stop("No overlapping item_uid between human_by_age and model")
  mean_kl_fun <- function(beta) {
    model_probs <- human_joined_opt %>%
      select(ends_with("_m")) %>%
      rename_with(~ sub("_m$", "", .x)) %>%
      softmax_images(beta)
    kls <- numeric(nrow(human_joined_opt))
    for (i in seq_len(nrow(human_joined_opt))) {
      p <- as.numeric(human_joined_opt[i, paste0(img_cols, "_h")])
      q <- as.numeric(model_probs[i, img_cols])
      kls[i] <- kl_one_row(p, q)
    }
    mean(kls, na.rm = TRUE)
  }
  res_opt <- nloptr::nloptr(
    x0 = 1, eval_f = mean_kl_fun, lb = 0.025, ub = 40,
    opts = list(algorithm = "NLOPT_GN_DIRECT_L", ftol_abs = 1e-4, maxeval = 200)
  )
  beta <- res_opt$solution

  # D_KL per (item_uid, age_bin): join human to model (one row per item_uid), then KL per row
  model_probs_beta <- softmax_images(model_wide %>% select(starts_with("image")), beta)
  model_probs_beta$item_uid <- model_wide$item_uid
  human_joined <- human %>%
    inner_join(model_probs_beta, by = "item_uid", suffix = c("_h", "_m"))
  d_kl_tbl <- human_joined %>%
    rowwise() %>%
    mutate(D_KL = kl_one_row(
      c_across(ends_with("_h")),
      c_across(ends_with("_m"))
    )) %>%
    ungroup() %>%
    select(item_uid, age_bin, D_KL) %>%
    mutate(task = task_id, model = model_id, .before = 1)

  # Accuracy per item_uid: model argmax vs correct option (index first, then fallback to trials)
  correct_tbl <- get_correct_option_per_item(version, project_root, model_wide$item_uid)
  if (is.null(correct_tbl) || all(is.na(correct_tbl$correct_option))) {
    correct_tbl <- get_correct_option_from_trials(task_id, version, project_root, model_wide$item_uid)
  } else if (any(is.na(correct_tbl$correct_option))) {
    fill <- get_correct_option_from_trials(task_id, version, project_root, model_wide$item_uid)
    if (!is.null(fill))
      correct_tbl <- correct_tbl %>%
        left_join(fill, by = "item_uid", suffix = c("", "_y")) %>%
        mutate(correct_option = coalesce(correct_option, correct_option_y)) %>%
        select(-any_of("correct_option_y"))
  }
  model_pred <- model_probs_beta %>%
    mutate(pred = max.col(as.matrix(select(., all_of(img_cols))))) %>%
    select(item_uid, pred)
  accuracy_tbl <- model_pred %>%
    left_join(correct_tbl, by = "item_uid") %>%
    mutate(correct = as.integer(pred == correct_option)) %>%
    select(item_uid, correct) %>%
    mutate(task = task_id, model = model_id, .before = 1)

  list(d_kl = d_kl_tbl, accuracy = accuracy_tbl, beta = beta)
}

# Main
if (is.na(task_id) || is.na(model_id)) {
  message("Usage: Rscript compare_levante.R --task TASK --model MODEL [--version VERSION] [--results-dir DIR] [--output-dir DIR] [--output-dkl CSV] [--output-accuracy CSV] [--project-root ROOT]")
  quit(save = "no", status = 0)
}

result <- tryCatch(
  compare_one(task_id, model_id, version, results_base, project_root),
  error = function(e) {
    message("Error: ", conditionMessage(e))
    quit(save = "no", status = 1)
  }
)

dir.create(dirname(output_dkl), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(output_acc), recursive = TRUE, showWarnings = FALSE)
readr::write_csv(result$d_kl, output_dkl)
readr::write_csv(result$accuracy, output_acc)
message("Wrote ", output_dkl, " (", nrow(result$d_kl), " rows)")
message("Wrote ", output_acc, " (", nrow(result$accuracy), " rows)")
message("Beta = ", round(result$beta, 4), "; mean accuracy = ", round(mean(result$accuracy$correct, na.rm = TRUE), 4))
