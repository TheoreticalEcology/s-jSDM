source("utils.R")

testthat::test_that("sjSDM_cv", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  
  library(sjSDM)
  
  sim = simulate_SDM(sites = 20L, species = 4L)
  X1 = sim$env_weights
  Y1 = sim$response
  
  sjSDM:::check_module()
  device = is_gpu_available()
  
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, device=device, sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, lambda_coef = 0.0, device=device, sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, lambda_coef = 0.0, alpha_cov = 0.1, device=device, sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune = "grid", lambda_coef = 0.0, alpha_cov = 0.1, lambda_cov = c(0.0, 0.1), alpha_coef = c(0.01,0.02), device=device, sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = linear(X1, ~0+X1), iter = 1L, CV = 2L, tune_steps = 3L, device=device , sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = linear(X1, ~0+X1:X2), iter = 1L, CV = 2L, tune_steps = 3L, device=device, sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = linear(X1, ~0+X1:X2),biotic = bioticStruct(df = 10L, on_diag = TRUE), iter = 1L, CV = 2L, tune_steps = 3L, device=device, sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = DNN(X1, ~0+X1:X2, hidden = c(5L, 5L)),biotic = bioticStruct(df = 10L, on_diag = TRUE), iter = 1L, CV = 2L, tune_steps = 3L, device=device, sampling=10L), NA)
  testthat::expect_error({model = sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 25L, device=device, sampling=10L)}, NA)
  testthat::expect_error(suppressWarnings(plot(model)), NA)
  testthat::expect_error(summary(model), NA)
  
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, lambda_coef = 0.0, n_cores = 2L, device=device, sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, lambda_coef = 0.0, alpha_cov = 0.1, n_cores = 2L, device=device, sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune = "grid", lambda_coef = 0.0, alpha_cov = 0.1, lambda_cov = c(0.0, 0.1), alpha_coef = c(0.01,0.02), n_cores = 2L, device=device, sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = linear(X1, ~0+X1), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L , device=device, sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = linear(X1, ~0+X1:X2), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = linear(X1, ~0+X1:X2),biotic = bioticStruct(df = 10L, on_diag = TRUE), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, env = DNN(X1, ~0+X1:X2, hidden = c(5L, 3L)),biotic = bioticStruct(df = 10L, on_diag = TRUE), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L), NA)
  testthat::expect_error({model = sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 25L, device=device, sampling=10L)}, NA)
  testthat::expect_error(suppressWarnings(plot(model)), NA)
  testthat::expect_error(summary(model), NA)
  
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, device=device, biotic = bioticStruct(inverse = TRUE), sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, device=device, biotic = bioticStruct(inverse = TRUE, on_diag = TRUE), sampling=10L), NA)
  
  
  SP = matrix(runif(nrow(X1)*2, -1, 1), nrow(X1), 2)
  testthat::expect_error(sjSDM_cv(Y1, X1,spatial = linear(SP), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L), NA)
  testthat::expect_error(sjSDM_cv(Y1, X1,spatial = linear(SP, ~0+X1:X2), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L), NA)
  testthat::expect_error({model = sjSDM_cv(Y1, X1,spatial = DNN(SP, ~0+X1:X2, hidden = c(5L, 3L)), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L)}, NA)
  #testthat::expect_error(suppressWarnings(plot(model)), NA)
  testthat::expect_error(summary(model), NA)  
  
  
  ### sjSDM.tune ###
  testthat::expect_error(sjSDM.tune(sjSDM_cv(Y1, X1,spatial = linear(SP), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error(sjSDM.tune(sjSDM_cv(Y1, X1,spatial = linear(SP), iter = 1L, family = binomial(),CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error(sjSDM.tune(sjSDM_cv(Y1, X1,spatial = linear(SP), iter = 1L, family = binomial(), control = sjSDMControl(), CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error(sjSDM.tune(sjSDM_cv(Y1, X1,spatial = linear(SP), CV = 2L, tune_steps = 3L, n_cores = 2L)), NA)
  
})
