# LEVANTE comparison: responses_by_ability (item_uid × ability_bin) vs model (one row per item_uid).
# D_KL per (item_uid, ability_bin); accuracy per item_uid with IRT d parameter.
# Usage: Rscript compare_levante.R --task TASK --model MODEL [--version VERSION] [--output-dir DIR] [--output-dkl CSV] [--output-accuracy CSV]
# Requires: tidyverse, philentropy, nloptr, reticulate

library(tidyverse)
library(reticulate)


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

data_raw     <- file.path(project_root, "data", "responses", version)
data_assets  <- file.path(project_root, "data", "assets", version)
results_base <- file.path(project_root, results_dir, version)

safe_task  <- gsub("[^a-zA-Z0-9_-]", "_", task_id)
safe_model <- gsub("[^a-zA-Z0-9_-]", "_", model_id)

if (is.na(output_dkl) || nchar(output_dkl) == 0L)
  output_dkl <- file.path(output_dir, paste0(safe_task, "_", safe_model, "_d_kl.csv"))
if (is.na(output_acc) || nchar(output_acc) == 0L)
  output_acc <- file.path(output_dir, paste0(safe_task, "_", safe_model, "_accuracy.csv"))

# ── Helpers ───────────────────────────────────────────────────────────────────

get_item_uid_order <- function(task_id, version, project_root) {
  tasks_dir   <- file.path(project_root, "data", "responses", version, "tasks")
  trials_path <- file.path(project_root, "data", "responses", version, "trials.csv")
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

get_responses_by_ability <- function(task_id, version, project_root) {
  resp_dir <- file.path(project_root, "data", "responses", version, "responses_by_ability")
  path <- file.path(resp_dir, paste0(safe_task, "_proportions_by_ability.csv"))
  if (!file.exists(path)) stop("Responses-by-ability file not found: ", path,
                                " (run download_levante_data.R first)")
  read_csv(path, show_col_types = FALSE)
}

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

load_item_difficulties <- function(task_id, version, project_root) {
  params_path <- file.path(project_root, "data", "responses", version, "irt_models",
                           paste0(safe_task, "_item_params.csv"))
  if (!file.exists(params_path)) return(NULL)
  readr::read_csv(params_path, show_col_types = FALSE) %>%
    select(item_uid, difficulty)
}

# The download script guarantees image1 = target (answer) for all items,
# so correct_option is always 1.

# ── Main comparison ───────────────────────────────────────────────────────────

compare_one <- function(task_id, model_id, version, results_base, project_root) {
  item_uid_order <- get_item_uid_order(task_id, version, project_root)
  human <- get_responses_by_ability(task_id, version, project_root)
  model_wide <- load_model_by_item(results_base, task_id, model_id, item_uid_order)

  n_opts <- length(grep("^image[0-9]+$", names(model_wide), value = TRUE))
  img_cols <- paste0("image", seq_len(n_opts))

  # β optimization: minimize mean KL over all (item_uid, ability_bin) pairs
  human_joined_opt <- human %>%
    inner_join(model_wide %>% select(item_uid, all_of(img_cols)), by = "item_uid", suffix = c("_h", "_m"))
  if (nrow(human_joined_opt) == 0L) stop("No overlapping item_uid between responses_by_ability and model")
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

  # D_KL per (item_uid, ability_bin)
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
    select(item_uid, ability_bin, D_KL) %>%
    mutate(task = task_id, model = model_id, .before = 1)

  # Accuracy per item_uid: model argmax vs image1 (= target, guaranteed by download script)
  model_pred <- model_probs_beta %>%
    mutate(pred = max.col(as.matrix(select(., all_of(img_cols))))) %>%
    select(item_uid, pred)

  # Join IRT item difficulties
  difficulties <- load_item_difficulties(task_id, version, project_root)

  accuracy_tbl <- model_pred %>%
    mutate(correct = as.integer(pred == 1L)) %>%
    select(item_uid, correct)
  if (!is.null(difficulties)) {
    accuracy_tbl <- accuracy_tbl %>% left_join(difficulties, by = "item_uid")
  } else {
    accuracy_tbl <- accuracy_tbl %>% mutate(difficulty = NA_real_)
  }
  accuracy_tbl <- accuracy_tbl %>%
    mutate(task = task_id, model = model_id, .before = 1)

  list(d_kl = d_kl_tbl, accuracy = accuracy_tbl, beta = beta)
}

# ── Entry point ───────────────────────────────────────────────────────────────

if (is.na(task_id) || is.na(model_id)) {
  message("Usage: Rscript compare_levante.R --task TASK --model MODEL [--version VERSION] ",
          "[--results-dir DIR] [--output-dir DIR] [--output-dkl CSV] [--output-accuracy CSV] ",
          "[--project-root ROOT]")
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
message("Beta = ", round(result$beta, 4),
        "; mean accuracy = ", round(mean(result$accuracy$correct, na.rm = TRUE), 4),
        "; IRT d/easiness correlation = ",
        tryCatch(round(cor(result$accuracy$correct, result$accuracy$difficulty,
                           use = "pairwise.complete.obs"), 4), error = function(e) "NA"))
