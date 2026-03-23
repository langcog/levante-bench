# LEVANTE comparison: softmax, KL, beta optimization, RSA (adapted from DevBench).
# Dependencies: tidyverse, philentropy, nloptr. Optional: reticulate (for reading .npy).

library(tidyverse)
library(philentropy)
library(nloptr)

## RSA functions
rsa <- function(mat1, mat2, method = "spearman") {
  mat1_lower <- mat1[lower.tri(mat1)]
  mat2_lower <- mat2[lower.tri(mat2)]
  cor(mat1_lower, mat2_lower, use = "pairwise.complete.obs", method = method)
}

get_permutations <- function(mat1, mat2, method = "spearman", nsim = 1000, seed = 42) {
  set.seed(seed)
  sims <- sapply(1:nsim, function(sim) {
    idx <- sample(nrow(mat1))
    mat1_perm <- mat1[idx, idx]
    rsa(mat1_perm, mat2, method = method)
  })
}

calc_permuted_p <- function(sim_cors, obs_cor) {
  sum(abs(obs_cor) < abs(sim_cors)) / length(sim_cors)
}

## Image (option) functions: forced-choice with columns image1..imageN or option1..optionN
option_cols <- function(data) {
  nms <- names(data)
  img <- nms[grepl("^image[0-9]+$", nms)]
  if (length(img) > 0) return(img)
  nms[grepl("^option[0-9]+$", nms)]
}

softmax_images <- function(data, beta = 1) {
  data %>%
    mutate(across(starts_with("image"), function(i) exp(beta * i)),
           rowsum = rowSums(across(starts_with("image"))),
           across(starts_with("image"), function(i) i / rowsum)) %>%
    select(-rowsum)
}

## Single-row KL: human (p) and model (q) as vectors; returns KL(p || q)
kl_one_row <- function(p, q) {
  p <- replace_na(as.numeric(p), 0)
  q <- replace_na(as.numeric(q), 0)
  if (length(p) != length(q)) return(NA_real_)
  m <- rbind(p, q)
  suppressWarnings(philentropy::KL(m, unit = "log"))
}

get_mean_kl_img <- function(human_probs_wide, model_probs_wide, return_distribs = FALSE) {
  combined_distribs <- bind_rows(human_probs_wide, model_probs_wide) %>%
    mutate(across(starts_with("image"), function(i) replace_na(i, 0))) %>%
    nest(distribs = -trial) %>%
    filter(vapply(distribs, nrow, integer(1)) == 2) %>%
    mutate(kl = sapply(distribs, function(d) {
      m <- d %>% select(starts_with("image")) %>% as.matrix()
      suppressWarnings(philentropy::KL(m, unit = "log"))
    }))
  if (return_distribs) return(combined_distribs)
  mean(combined_distribs$kl, na.rm = TRUE)
}

## Text functions (for text-based tasks)
softmax_texts <- function(data, beta = 1) {
  data %>%
    group_by(cue) %>%
    mutate(model = exp(beta * model),
           model_sum = sum(model),
           model = model / model_sum) %>%
    select(-model_sum)
}

get_mean_kl_txt <- function(combined_probs) {
  combined_distribs <- combined_probs %>%
    select(-target) %>%
    nest(distribs = -cue) %>%
    mutate(distribs = lapply(distribs, function(d) t(d)),
           kl = sapply(distribs, function(d) {
             suppressWarnings(philentropy::KL(as.matrix(d), unit = "log"))
           }))
  mean(combined_distribs$kl, na.rm = TRUE)
}

## Optimization: find beta that minimizes mean KL (image format)
get_opt_kl <- function(human_probs_wide, model_logits_wide) {
  mean_kl <- function(beta) {
    get_mean_kl_img(human_probs_wide, softmax_images(model_logits_wide, beta))
  }
  res <- nloptr::nloptr(x0 = 1,
                        eval_f = mean_kl,
                        lb = 0.025,
                        ub = 40,
                        opts = list(algorithm = "NLOPT_GN_DIRECT_L",
                                    ftol_abs = 1e-4,
                                    maxeval = 200))
  list(objective = res$objective, solution = res$solution, iterations = res$iterations)
}

get_opt_kl_txt <- function(combined_data) {
  mean_kl <- function(beta) get_mean_kl_txt(softmax_texts(combined_data, beta))
  res <- nloptr::nloptr(x0 = 1,
                        eval_f = mean_kl,
                        lb = 0.025,
                        ub = 40,
                        opts = list(algorithm = "NLOPT_GN_DIRECT_L",
                                    ftol_abs = 1e-4,
                                    maxeval = 200))
  list(objective = res$objective, solution = res$solution, iterations = res$iterations)
}

# Fixed beta (no optimization)
get_reg_kl <- function(human_probs_wide, model_logits_wide, beta = 1) {
  res <- get_mean_kl_img(human_probs_wide, softmax_images(model_logits_wide, beta))
  list(objective = res, solution = beta, iterations = 0)
}

get_opt_kl_allage <- function(human_data, model_logits_wide, beta) {
  human_data %>%
    select(age_bin, trial, starts_with("image")) %>%
    nest(data = -age_bin) %>%
    mutate(kl = sapply(data, function(d) get_mean_kl_img(d, softmax_images(model_logits_wide, beta)))) %>%
    pull(kl) %>%
    mean(na.rm = TRUE)
}
