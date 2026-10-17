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
  
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, device=device, sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, lambda_coef = 0.0, device=device, sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, lambda_coef = 0.0, alpha_cov = 0.1, device=device, sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune = "grid", lambda_coef = 0.0, alpha_cov = 0.1, lambda_cov = c(0.0, 0.1), alpha_coef = c(0.01,0.02), device=device, sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, env = linear(X1, ~0+X1), iter = 1L, CV = 2L, tune_steps = 3L, device=device , sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, env = linear(X1, ~0+X1:X2), iter = 1L, CV = 2L, tune_steps = 3L, device=device, sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, env = linear(X1, ~0+X1:X2),biotic = bioticStruct(df = 10L, on_diag = TRUE), iter = 1L, CV = 2L, tune_steps = 3L, device=device, sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, env = DNN(X1, ~0+X1:X2, hidden = c(5L, 5L)),biotic = bioticStruct(df = 10L, on_diag = TRUE), iter = 1L, CV = 2L, tune_steps = 3L, device=device, sampling=10L)), NA)
  testthat::expect_error({model = suppressWarnings(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 25L, device=device, sampling=10L))}, NA)
  testthat::expect_error(suppressWarnings(plot(model)), NA)
  testthat::expect_error(summary(model), NA)
  
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, lambda_coef = 0.0, n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, lambda_coef = 0.0, alpha_cov = 0.1, n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune = "grid", lambda_coef = 0.0, alpha_cov = 0.1, lambda_cov = c(0.0, 0.1), alpha_coef = c(0.01,0.02), n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, env = linear(X1, ~0+X1), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L , device=device, sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, env = linear(X1, ~0+X1:X2), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, env = linear(X1, ~0+X1:X2),biotic = bioticStruct(df = 10L, on_diag = TRUE), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, env = DNN(X1, ~0+X1:X2, hidden = c(5L, 3L)),biotic = bioticStruct(df = 10L, on_diag = TRUE), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error({model = suppressWarnings(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 25L, device=device, sampling=10L))}, NA)
  testthat::expect_error(suppressWarnings(plot(model)), NA)
  testthat::expect_error(summary(model), NA)
  
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, device=device, biotic = bioticStruct(inverse = TRUE), sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, X1, iter = 1L, CV = 2L, tune_steps = 3L, device=device, biotic = bioticStruct(inverse = TRUE, on_diag = TRUE), sampling=10L)), NA)
  
  
  SP = matrix(runif(nrow(X1)*2, -1, 1), nrow(X1), 2)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, X1,spatial = linear(SP), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error(suppressWarnings(sjSDM_cv(Y1, X1,spatial = linear(SP, ~0+X1:X2), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error({model = suppressWarnings(sjSDM_cv(Y1, X1,spatial = DNN(SP, ~0+X1:X2, hidden = c(5L, 3L)), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L))}, NA)
  #testthat::expect_error(suppressWarnings(plot(model)), NA)
  testthat::expect_error(summary(model), NA)  
  
  
  ### sjSDM.tune ###
  testthat::expect_error(sjSDM.tune(sjSDM_cv(Y1, X1,spatial = linear(SP), iter = 1L, CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error(sjSDM.tune(sjSDM_cv(Y1, X1,spatial = linear(SP), iter = 1L, family = binomial(),CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error(sjSDM.tune(sjSDM_cv(Y1, X1,spatial = linear(SP), iter = 1L, family = binomial(), control = sjSDMControl(), CV = 2L, tune_steps = 3L, n_cores = 2L, device=device, sampling=10L)), NA)
  testthat::expect_error(sjSDM.tune(sjSDM_cv(Y1, X1,spatial = linear(SP), CV = 2L, tune_steps = 3L, n_cores = 2L)), NA)
  
})


# The per-fold refit rebuilds the design from data.frame(X), which renames "(Intercept)".
# Re-deriving the penalty flag from that column name silently penalised the intercept, twice.
# Pinned against the equivalent direct fit: the two are bit-identical once the flag is carried.
testthat::test_that("sjSDM_cv folds keep the intercept out of the env penalty", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(42)
  sim = simulate_SDM(sites = 60L, species = 5L)
  X1 = sim$env_weights
  Y1 = sim$response

  cv = suppressWarnings(sjSDM_cv(Y1, X1, tune = "grid", CV = 2L, iter = 5L, sampling = 10L,
                                 lambda_coef = 0.5, alpha_coef = 0.5, lambda_cov = 0.0,
                                 alpha_cov = 0.5, device = "cpu"))
  fold = cv$tune_results[[1]][[1]]
  train = setdiff(seq_len(nrow(X1)), fold$indices)
  direct = sjSDM(Y1[train, , drop = FALSE],
                 env = linear(X1[train, , drop = FALSE], ~., lambda = 0.5, alpha = 0.5),
                 iter = 5L, sampling = 10L, device = "cpu", verbose = FALSE)

  testthat::expect_equal(fold$ll_train, direct$logLik[[1]], tolerance = 1e-5)
})
