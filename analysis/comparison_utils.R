# Comparison Analysis Utilities for PENPAL-COLING
# Standardized data loading, normalization, and Linear Mixed-Effects Models (LMM) for computed metrics
# 
# Required R packages loaded at top of file
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, here, lme4, lmerTest, emmeans, effectsize, patchwork)

# Project root directory
PROJECT_ROOT <- here::here()

# =============================================================================
# 1. Condition & Metric Domain Definitions
# =============================================================================

# Canonical condition levels and labels
CONDITION_LEVELS <- c("HH", "H-LLM", "LLM-LLM")
CONDITION_LABELS <- c(
  "HH" = "Human-Human",
  "H-LLM" = "Human-AI",
  "LLM-LLM" = "AI-AI"
)

# Color palette for conditions
condition_colors <- c(
  "HH" = "#2ca02c",          # Green for Human-Human
  "H-LLM" = "#1f77b4",       # Blue for Human-AI
  "LLM-LLM" = "#ff7f0e",     # Orange for AI-AI
  "Human-Human" = "#2ca02c",
  "Human-AI" = "#1f77b4",
  "AI-AI" = "#ff7f0e"
)

# Metric family definitions matching story_metrics.csv columns
RQA_RR_METRICS <- c(
  "rr_RR", "rr_DET", "rr_L", "rr_L_MAX", "rr_DIV",
  "rr_ENTR", "rr_RATIO", "rr_LAM", "rr_TT", "rr_V_MAX"
)

RQA_EPS_METRICS <- c(
  "eps_RR", "eps_DET", "eps_L", "eps_L_MAX", "eps_DIV",
  "eps_ENTR", "eps_RATIO", "eps_LAM", "eps_TT", "eps_V_MAX", "eps_story"
)

RQA_METRICS <- c(RQA_RR_METRICS, RQA_EPS_METRICS)

TRAJECTORY_METRICS <- c(
  "path_length", "mean_step", "sd_step", "max_step",
  "net_displacement", "straightness", "radius_gyration",
  "centroid_norm", "msd_alpha", "msd_alpha_r2", "circularity",
  "first_last_cos", "mean_pairwise_sim", "sd_pairwise_sim"
)

DECAY_METRICS <- c(
  "decay_lambda", "decay_A", "decay_C", "decay_exp_r2",
  "decay_half_life", "decay_fit_at_bound", "decay_beta",
  "decay_pow_r2", "decay_lin_slope", "decay_lin_r2",
  "decay_half_distance", "sim_lag1", "sim_maxlag"
)

HUMAN_RATING_METRICS <- c(
  "overall_coherence", "overall_creativity", "overall_likeability",
  "mean_coherence_element_consistency", "mean_coherence_logical_progression",
  "mean_creativity_originality", "mean_creativity_surprisingness",
  "mean_likeability_enjoyability", "mean_likeability_quality", "mean_lead_time"
)

# =============================================================================
# 2. Data Cleaning & Normalization Helpers
# =============================================================================

#' Normalize condition identifier to canonical level ("HH", "H-LLM", "LLM-LLM")
normalize_condition_id <- function(x) {
  x <- as.character(x)
  case_when(
    str_detect(tolower(x), "^hh$|human-human|human_human") ~ "HH",
    str_detect(tolower(x), "^ha$|^h-llm$|human-ai|human_ai|h_llm") ~ "H-LLM",
    str_detect(tolower(x), "^aa$|^llm-llm$|ai-ai|ai_ai|llm_llm") ~ "LLM-LLM",
    TRUE ~ x
  )
}

#' Prepare condition column as an ordered factor
prepare_condition_factor <- function(df, reference = "HH", use_long_labels = FALSE) {
  cond_col <- if ("cond" %in% names(df)) "cond" else if ("condition" %in% names(df)) "condition" else NULL
  if (is.null(cond_col)) return(df)

  canonical_ref <- normalize_condition_id(reference)
  levels_order <- c(canonical_ref, setdiff(CONDITION_LEVELS, canonical_ref))

  df <- df %>%
    mutate(
      cond = normalize_condition_id(.data[[cond_col]]),
      condition = factor(
        cond,
        levels = levels_order,
        labels = if (use_long_labels) CONDITION_LABELS[levels_order] else levels_order
      )
    )
  df
}

# =============================================================================
# 3. Data Loading Functions
# =============================================================================

#' Load computed story-level metrics from story_metrics.csv
load_story_metrics <- function(file_path = NULL, reference_cond = "HH", use_long_labels = FALSE) {
  path <- if (is.null(file_path)) file.path(PROJECT_ROOT, "data", "processed", "rqa", "story_metrics.csv") else file_path

  if (!file.exists(path)) {
    stop(paste("story_metrics.csv not found at:", path))
  }

  df <- read_csv(path, show_col_types = FALSE) %>%
    mutate(
      story_id = as.character(story_id),
      conversation_id = if ("conversation_id" %in% names(.)) as.character(conversation_id) else story_id
    ) %>%
    prepare_condition_factor(reference = reference_cond, use_long_labels = use_long_labels)

  df
}

#' Load lag profiles from lag_profiles.csv
load_lag_profiles <- function(file_path = NULL, story_metrics_df = NULL) {
  path <- if (is.null(file_path)) file.path(PROJECT_ROOT, "data", "processed", "rqa", "lag_profiles.csv") else file_path

  if (!file.exists(path)) {
    stop(paste("lag_profiles.csv not found at:", path))
  }

  df <- read_csv(path, show_col_types = FALSE) %>%
    mutate(story_id = as.character(story_id))

  if (!is.null(story_metrics_df) && "story_id" %in% names(story_metrics_df)) {
    meta <- story_metrics_df %>% select(story_id, cond, condition) %>% distinct()
    df <- df %>% left_join(meta, by = "story_id")
  } else if (file.exists(file.path(PROJECT_ROOT, "data", "processed", "rqa", "story_metrics.csv"))) {
    meta <- load_story_metrics() %>% select(story_id, cond, condition) %>% distinct()
    df <- df %>% left_join(meta, by = "story_id")
  }

  df
}

#' Load sentence index mapping from sentence_index.csv
load_sentence_index <- function(file_path = NULL) {
  path <- if (is.null(file_path)) file.path(PROJECT_ROOT, "data", "interim", "embeddings", "sentence_index.csv") else file_path

  if (!file.exists(path)) {
    stop(paste("sentence_index.csv not found at:", path))
  }

  read_csv(path, show_col_types = FALSE) %>%
    mutate(story_id = as.character(story_id))
}

#' Load story metadata and annotations from stories.csv
load_stories_data <- function(file_path = NULL, reference_cond = "HH") {
  path <- if (is.null(file_path)) file.path(PROJECT_ROOT, "data", "interim", "embeddings", "stories.csv") else file_path

  if (!file.exists(path)) {
    stop(paste("stories.csv not found at:", path))
  }

  read_csv(path, show_col_types = FALSE) %>%
    mutate(story_id = as.character(story_id)) %>%
    prepare_condition_factor(reference = reference_cond)
}

# =============================================================================
# 4. Summary Statistics Helpers
# =============================================================================

#' Compute descriptive statistics by condition for a specific metric
summarize_by_condition <- function(df, var, group_vars = "condition") {
  df %>%
    filter(!is.na(.data[[var]])) %>%
    group_by(across(all_of(group_vars))) %>%
    summarize(
      n = n(),
      mean = mean(.data[[var]], na.rm = TRUE),
      sd = sd(.data[[var]], na.rm = TRUE),
      se = sd / sqrt(n),
      ci_lower = mean - 1.96 * se,
      ci_upper = mean + 1.96 * se,
      median = median(.data[[var]], na.rm = TRUE),
      iqr = IQR(.data[[var]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(metric = var)
}

#' Summarize multiple metrics across conditions into a single tidy dataframe
summarize_all_metrics <- function(df, metrics_vector, group_vars = "condition") {
  valid_metrics <- intersect(metrics_vector, names(df))
  if (length(valid_metrics) == 0) {
    stop("None of the specified metrics exist in the input dataframe.")
  }

  map_dfr(valid_metrics, ~ summarize_by_condition(df, var = .x, group_vars = group_vars))
}

# =============================================================================
# 5. Mixed Model Fitting & Post-Hoc Analysis (lmerTest / lme4)
# =============================================================================

#' Fit Linear Mixed Model (or Linear Model via stats::lm) comparing conditions
#'
#' @param df Data frame containing the target metric and condition factor
#' @param var Character: Dependent variable name
#' @param random_effects Character: Random effects syntax e.g. "(1 | conversation_id)"
#' @param covariates Character vector: Additional covariates to control for e.g. c("n_words")
#' @param group_var Character: Condition variable name (default: "condition")
#' @param REML Logical: Use Restricted Maximum Likelihood (default: TRUE)
#' @return An lmerTest object (or lm object if no random effects)
fit_condition_lmm <- function(df, var, random_effects = NULL, covariates = NULL, group_var = "condition", REML = TRUE) {
  sub_df <- df %>% filter(!is.na(.data[[var]]), !is.na(.data[[group_var]]))
  
  # Build formula terms
  fixed_terms <- c(group_var, covariates)
  fixed_part <- paste(fixed_terms, collapse = " + ")

  if (!is.null(random_effects) && nchar(random_effects) > 0) {
    formula_str <- paste(var, "~", fixed_part, "+", random_effects)
    model <- lmerTest::lmer(
      as.formula(formula_str),
      data = sub_df,
      REML = REML,
      control = lme4::lmerControl(optimizer = "bobyqa")
    )
  } else {
    formula_str <- paste(var, "~", fixed_part)
    model <- stats::lm(as.formula(formula_str), data = sub_df)
  }

  model
}

#' Fit LMM and extract Type III ANOVA table (Satterthwaite df) & Effect Sizes
#'
#' @param df Data frame
#' @param var Character: Metric variable name
#' @param random_effects Character: Random effects specification
#' @param covariates Character vector: Covariate names
#' @param group_var Character: Condition column name
#' @return Tidy tibble containing F-statistic, NumDF, DenDF, p-value, and Partial Eta Squared
compare_conditions_lmm <- function(df, var, random_effects = NULL, covariates = NULL, group_var = "condition") {
  sub_df <- df %>% filter(!is.na(.data[[var]]), !is.na(.data[[group_var]]))
  if (nrow(sub_df) == 0) return(NULL)

  model <- fit_condition_lmm(
    df = sub_df,
    var = var,
    random_effects = random_effects,
    covariates = covariates,
    group_var = group_var
  )

  # Extract joint Type III F-tests via emmeans::joint_tests or lmerTest::anova
  j_tests <- tryCatch(
    emmeans::joint_tests(model),
    error = function(e) NULL
  )

  if (!is.null(j_tests) && "model term" %in% names(j_tests)) {
    match_row <- j_tests %>% filter(`model term` == group_var)
    if (nrow(match_row) > 0) {
      f_stat <- match_row$F.ratio[1]
      num_df <- match_row$df1[1]
      den_df <- match_row$df2[1]
      p_val <- match_row$p.value[1]
    } else {
      f_stat <- NA_real_; num_df <- NA_real_; den_df <- NA_real_; p_val <- NA_real_
    }
  } else {
    aov_tab <- tryCatch(anova(model), error = function(e) NULL)
    if (!is.null(aov_tab) && "F value" %in% names(aov_tab)) {
      f_stat <- aov_tab[group_var, "F value"]
      num_df <- if ("NumDF" %in% names(aov_tab)) aov_tab[group_var, "NumDF"] else aov_tab[group_var, "Df"]
      den_df <- if ("DenDF" %in% names(aov_tab)) aov_tab[group_var, "DenDF"] else df.residual(model)
      p_val <- if ("Pr(>F)" %in% names(aov_tab)) aov_tab[group_var, "Pr(>F)"] else NA_real_
    } else {
      f_stat <- NA_real_; num_df <- NA_real_; den_df <- NA_real_; p_val <- NA_real_
    }
  }

  # Partial Eta Squared
  pes <- tryCatch(
    effectsize::eta_squared(model, partial = TRUE, verbose = FALSE)$Eta2_partial[1],
    error = function(e) NA_real_
  )

  tibble(
    metric = var,
    n = nrow(sub_df),
    F_stat = f_stat,
    NumDF = num_df,
    DenDF = den_df,
    p_value = p_val,
    partial_eta_sq = pes
  )
}

#' Pairwise post-hoc contrasts using estimated marginal means (emmeans) with Bonferroni adjustment
get_condition_contrasts <- function(model_or_df, var = NULL, random_effects = NULL, covariates = NULL, group_var = "condition", adjust = "bonferroni") {
  fit <- if (is.data.frame(model_or_df)) {
    if (is.null(var)) stop("Must specify `var` when passing a dataframe to get_condition_contrasts.")
    fit_condition_lmm(model_or_df, var = var, random_effects = random_effects, covariates = covariates, group_var = group_var)
  } else {
    model_or_df
  }

  emm <- emmeans::emmeans(fit, specs = group_var)
  pairs_out <- contrast(emm, method = "pairwise", adjust = adjust)
  
  as_tibble(pairs_out)
}

#' Single-df planned contrast: Mixed condition (Human-AI) vs Average of Same-type conditions (HH + LLM-LLM)/2
planned_contrast_HA_vs_same <- function(df, value_col, ha_label = "H-LLM", hh_label = "HH", aa_label = "LLM-LLM", group_var = "cond") {
  d <- df %>% filter(!is.na(.data[[value_col]]), !is.na(.data[[group_var]]))
  
  ha_vals <- d[[value_col]][as.character(d[[group_var]]) == ha_label]
  hh_vals <- d[[value_col]][as.character(d[[group_var]]) == hh_label]
  aa_vals <- d[[value_col]][as.character(d[[group_var]]) == aa_label]

  if (length(ha_vals) < 2 || length(hh_vals) < 2 || length(aa_vals) < 2) {
    return(tibble(contrast = "HA vs (HH+AA)/2", estimate = NA_real_, se = NA_real_, t = NA_real_, df = NA_real_, p = NA_real_))
  }

  m_ha <- mean(ha_vals); v_ha <- var(ha_vals); n_ha <- length(ha_vals)
  m_hh <- mean(hh_vals); v_hh <- var(hh_vals); n_hh <- length(hh_vals)
  m_aa <- mean(aa_vals); v_aa <- var(aa_vals); n_aa <- length(aa_vals)

  est <- m_ha - 0.5 * (m_hh + m_aa)
  se <- sqrt(v_ha / n_ha + 0.25 * v_hh / n_hh + 0.25 * v_aa / n_aa)
  t_stat <- est / se

  df_welch <- (v_ha / n_ha + 0.25 * v_hh / n_hh + 0.25 * v_aa / n_aa)^2 /
    ((v_ha / n_ha)^2 / (n_ha - 1) + (0.25 * v_hh / n_hh)^2 / (n_hh - 1) + (0.25 * v_aa / n_aa)^2 / (n_aa - 1))
  
  p_val <- 2 * pt(-abs(t_stat), df = df_welch)

  tibble(
    contrast = sprintf("%s vs (%s + %s)/2", ha_label, hh_label, aa_label),
    estimate = est,
    se = se,
    t = t_stat,
    df = df_welch,
    p = p_val,
    ci_lower = est - qt(0.975, df_welch) * se,
    ci_upper = est + qt(0.975, df_welch) * se
  )
}

#' Compute pairwise non-parametric effect sizes (Cliff's delta and Cohen's d)
effect_sizes_pairwise <- function(df, value_col, group_var = "cond") {
  conds <- sort(unique(as.character(df[[group_var]])))
  pairs_list <- combn(conds, 2, simplify = FALSE)

  cliff_delta_calc <- function(x, y) {
    x <- x[!is.na(x)]; y <- y[!is.na(y)]
    if (length(x) == 0 || length(y) == 0) return(NA_real_)
    (sum(outer(x, y, ">")) - sum(outer(x, y, "<"))) / (length(x) * length(y))
  }

  cohens_d_calc <- function(x, y) {
    x <- x[!is.na(x)]; y <- y[!is.na(y)]
    if (length(x) < 2 || length(y) < 2) return(NA_real_)
    s_pooled <- sqrt(((length(x) - 1) * var(x) + (length(y) - 1) * var(y)) / (length(x) + length(y) - 2))
    (mean(x) - mean(y)) / s_pooled
  }

  map_dfr(pairs_list, function(p) {
    a <- df[[value_col]][as.character(df[[group_var]]) == p[[1]]]
    b <- df[[value_col]][as.character(df[[group_var]]) == p[[2]]]

    c_delta <- cliff_delta_calc(a, b)
    c_d <- cohens_d_calc(a, b)

    tibble(
      contrast = paste(p[[1]], "vs", p[[2]]),
      n_1 = sum(!is.na(a)),
      n_2 = sum(!is.na(b)),
      cohens_d = c_d,
      cliff_delta = c_delta,
      magnitude = case_when(
        is.na(c_delta) ~ NA_character_,
        abs(c_delta) < 0.147 ~ "negligible",
        abs(c_delta) < 0.330 ~ "small",
        abs(c_delta) < 0.474 ~ "medium",
        TRUE ~ "large"
      )
    )
  })
}

#' Bootstrap confidence intervals for condition means
bootstrap_condition_means <- function(df, value_col, group_var = "cond", R = 10000, ci = 0.95, seed = 42) {
  set.seed(seed)
  alpha <- (1 - ci) / 2

  df %>%
    filter(!is.na(.data[[value_col]]), !is.na(.data[[group_var]])) %>%
    group_by(across(all_of(group_var))) %>%
    group_modify(function(g, key) {
      x <- g[[value_col]]
      n <- length(x)
      if (n < 2) {
        return(tibble(n = n, mean = mean(x), ci_lower = NA_real_, ci_upper = NA_real_))
      }
      boots <- replicate(R, mean(sample(x, n, replace = TRUE)))
      tibble(
        n = n,
        mean = mean(x),
        ci_lower = unname(quantile(boots, alpha)),
        ci_upper = unname(quantile(boots, 1 - alpha))
      )
    }) %>%
    ungroup()
}

#' Spearman/Pearson correlation between a target computed metric and human ratings
correlate_metric_with_ratings <- function(df, metric_col, rating_cols = HUMAN_RATING_METRICS, method = "spearman") {
  valid_ratings <- intersect(rating_cols, names(df))

  if (!metric_col %in% names(df) || length(valid_ratings) == 0) {
    stop("Metric or rating columns not found in dataframe.")
  }

  map_dfr(valid_ratings, function(r_col) {
    sub_df <- df %>% filter(!is.na(.data[[metric_col]]), !is.na(.data[[r_col]]))
    if (nrow(sub_df) < 3) {
      return(tibble(metric = metric_col, rating = r_col, n = nrow(sub_df), r = NA_real_, p_value = NA_real_))
    }
    ct <- cor.test(sub_df[[metric_col]], sub_df[[r_col]], method = method)
    tibble(
      metric = metric_col,
      rating = r_col,
      n = nrow(sub_df),
      r = unname(ct$estimate),
      p_value = ct$p.value,
      method = method
    )
  })
}

# =============================================================================
# 6. Plotting Utilities & Themes
# =============================================================================

#' Standard publication-ready ggplot2 theme
theme_comparison <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "#e0e0e0", fill = NA, linewidth = 0.5),
      strip.text = element_text(face = "bold", size = rel(1.05)),
      plot.title = element_text(hjust = 0.5, face = "bold", size = rel(1.1)),
      plot.subtitle = element_text(hjust = 0.5, color = "grey30", size = rel(0.95)),
      axis.title = element_text(face = "bold")
    )
}

#' Bar plot with 95% confidence intervals comparing conditions
plot_condition_comparison <- function(summary_df, title = "", ylab = "Value", metric_name = NULL) {
  if (!is.null(metric_name) && "metric" %in% names(summary_df)) {
    summary_df <- summary_df %>% filter(metric == metric_name)
  }

  group_col <- if ("condition" %in% names(summary_df)) "condition" else "cond"

  ggplot(summary_df, aes(x = .data[[group_col]], y = mean, fill = .data[[group_col]])) +
    geom_col(width = 0.65, alpha = 0.85, color = "black", linewidth = 0.3) +
    geom_errorbar(
      aes(ymin = ci_lower, ymax = ci_upper),
      width = 0.2,
      linewidth = 0.8,
      color = "#333333"
    ) +
    scale_fill_manual(values = condition_colors) +
    labs(title = title, y = ylab, x = "Condition") +
    theme_comparison() +
    theme(legend.position = "none")
}

#' Boxplot with jittered data points for metric distributions across conditions
plot_metric_boxplots <- function(df, var, title = "", ylab = NULL) {
  group_col <- if ("condition" %in% names(df)) "condition" else "cond"
  y_label <- if (is.null(ylab)) var else ylab

  ggplot(df, aes(x = .data[[group_col]], y = .data[[var]], fill = .data[[group_col]])) +
    geom_boxplot(outlier.shape = NA, alpha = 0.7, width = 0.5, color = "#222222") +
    geom_jitter(width = 0.15, alpha = 0.4, size = 1.8, aes(color = .data[[group_col]])) +
    scale_fill_manual(values = condition_colors) +
    scale_color_manual(values = condition_colors) +
    labs(title = title, y = y_label, x = "Condition") +
    theme_comparison() +
    theme(legend.position = "none")
}

#' Scatterplot with regression lines showing metric correlation with human rating
plot_metric_correlations <- function(df, metric_col, rating_col, title = NULL) {
  group_col <- if ("condition" %in% names(df)) "condition" else "cond"
  plot_title <- if (is.null(title)) paste(metric_col, "vs", rating_col) else title

  ggplot(df, aes(x = .data[[metric_col]], y = .data[[rating_col]])) +
    geom_point(aes(color = .data[[group_col]]), alpha = 0.6, size = 2) +
    geom_smooth(method = "lm", se = TRUE, color = "black", linewidth = 0.9, fill = "grey70") +
    scale_color_manual(values = condition_colors) +
    labs(title = plot_title, x = metric_col, y = rating_col, color = "Condition") +
    theme_comparison()
}

# =============================================================================
# 7. Data Availability Diagnostics
# =============================================================================

#' Check existence of project outputs and print health check table
check_data_availability <- function() {
  expected_files <- c(
    "data/processed/rqa/story_metrics.csv",
    "data/interim/embeddings/stories.csv",
    "data/processed/rqa/lag_profiles.csv",
    "data/interim/embeddings/sentence_index.csv",
    "data/processed/rqa/distance_matrices.npz",
    "data/processed/rqa/recurrence_matrices_rr.npz",
    "data/processed/rqa/recurrence_matrices_eps.npz",
    "data/processed/rqa/similarity_matrices.npz",
    "data/interim/annotations/penpal_annotations_final.csv"
  )

  results <- tibble(
    file = expected_files,
    exists = map_lgl(expected_files, ~ file.exists(file.path(PROJECT_ROOT, .x))),
    full_path = map_chr(expected_files, ~ file.path(PROJECT_ROOT, .x))
  )

  results
}

# Print status message when sourced
if (interactive()) {
  cat("\n=== PENPAL-COLING Comparison Utilities Loaded (Linear Mixed Models) ===\n")
  cat("Project root:", PROJECT_ROOT, "\n")
  cat("Run `check_data_availability()` to verify dataset output paths.\n\n")
}
