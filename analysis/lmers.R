library("tidyverse")
library("lme4")
library("lmerTest")

# NOTE: Set your working directory to the root of this repo (lm-task-demands)

# Helper function for fitting (G)LMER model for model size regressions.
fit_size_model <- function(task_name, binary_outcome = TRUE) {
  # Read and process data.
  df <- read.csv(sprintf("analysis/processed_data/%s_long_size.csv", task_name))
  df$model_family <- factor(df$model_family)
  df$eval_method <- factor(df$eval_method)
  df$model_n_params_log <- log10(df$model_n_params)
  
  if (binary_outcome) {
    df$response_correct <- ifelse(df$response_correct == "True", 1, 0)
    m <- glmer(
      response_correct ~ model_n_params_log*eval_method + (model_n_params_log*eval_method | model_family),
      data=df,
      family=binomial(link="logit")
    )
  }
  else {
    m <- lmer(
      logprob ~ model_n_params_log*eval_method + (model_n_params_log*eval_method | model_family),
      data=df
    )
  }
  
  # Save summary and model file.
  sink(sprintf("analysis/lmers/%s_lmer_size.txt", task_name))
  print(summary(m))
  sink()
  saveRDS(m, file=sprintf("analysis/lmers/%s_lmer_size.rds", task_name))
  return(m)
}

# Helper function for fitting (G)LM model for training time regressions.
fit_training_time_model <- function(task_name, binary_outcome = TRUE) {
  # Read and process data.
  df <- read.csv(sprintf("analysis/processed_data/%s_long_training_time.csv", task_name))
  df$eval_method <- factor(df$eval_method)
  df$training_step_log <- log10(df$training_step)
  
  if (binary_outcome) {
    df$response_correct <- ifelse(df$response_correct == "True", 1, 0)
    m <- glm(
      response_correct ~ training_step_log*eval_method,
      data=df,
      family=binomial(link="logit")
    )
  }
  else {
    m <- lm(
      logprob ~ training_step_log*eval_method,
      data=df
    )
  }
  
  # Save summary and model file.
  sink(sprintf("analysis/lmers/%s_lmer_training_time.txt", task_name))
  print(summary(m))
  sink()
  saveRDS(m, file=sprintf("analysis/lmers/%s_lmer_training_time.rds", task_name))
  return(m)
}

# Fit size models.
m.digit_mat.size <- fit_size_model("digit_mat", binary_outcome=TRUE)
m.crt.size <- fit_size_model("crt", binary_outcome=TRUE)
m.lambada.size <- fit_size_model("lambada", binary_outcome=FALSE)
m.syntax.size <- fit_size_model("syntax", binary_outcome=TRUE)

# Fit training time models.
m.digit_mat.train <- fit_training_time_model("digit_mat", binary_outcome=TRUE)
m.lambada.train <- fit_training_time_model("lambada", binary_outcome=FALSE)
