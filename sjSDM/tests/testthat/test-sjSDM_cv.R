context("sjSDM_cv")

source("utils.R")

testthat::test_that("sjSDM_cv", {
  skip_if_no_torch()
  
  library(sjSDM)
  
  sim = simulate_SDM(sites = 50L, species = 11L)
  X1 = sim$env_weights
  Y1 = sim$response
  
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L), NA)
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, lambda_coef = 0.0), NA)
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, lambda_coef = 0.0, alpha_cov = 0.1), NA)
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune = "grid", lambda_coef = 0.0, alpha_cov = 0.1, lambda_cov = c(0.0, 0.1), alpha_coef = c(0.01,0.02)), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = envLinear(X1, ~0+X1), iter = 1L, CV = 2L, tune_steps = 3L, ), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = envLinear(X1, ~0+X1:X2), iter = 1L, CV = 2L, tune_steps = 3L), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = envLinear(X1, ~0+X1:X2),biotic = bioticStruct(df = 10L, on_diag = TRUE), iter = 1L, CV = 2L, tune_steps = 3L), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = envDNN(X1, ~0+X1:X2, hidden = c(5L)),biotic = bioticStruct(df = 10L, on_diag = TRUE), iter = 1L, CV = 2L, tune_steps = 3L), NA)
  testthat::expect_error({model = sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 25L)}, NA)
  testthat::expect_error(suppressWarnings(plot(model)), NA)
  testthat::expect_error(summary(model), NA)
  
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L), NA)
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, lambda_coef = 0.0, n_cores = 2L), NA)
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, lambda_coef = 0.0, alpha_cov = 0.1, n_cores = 2L), NA)
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune = "grid", lambda_coef = 0.0, alpha_cov = 0.1, lambda_cov = c(0.0, 0.1), alpha_coef = c(0.01,0.02), n_cores = 2L), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = envLinear(X1, ~0+X1), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L ), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = envLinear(X1, ~0+X1:X2), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = envLinear(X1, ~0+X1:X2),biotic = bioticStruct(df = 10L, on_diag = TRUE), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = envDNN(X1, ~0+X1:X2, hidden = c(5L)),biotic = bioticStruct(df = 10L, on_diag = TRUE), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L), NA)
  testthat::expect_error({model = sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 25L)}, NA)
  testthat::expect_error(suppressWarnings(plot(model)), NA)
  testthat::expect_error(summary(model), NA)
  
})