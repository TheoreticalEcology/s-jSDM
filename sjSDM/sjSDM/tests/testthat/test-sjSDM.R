context("sjSDM")

source("utils.R")

testthat::test_that("sjSDM functionality", {
  skip_if_no_torch()

  sim = simulate_SDM(sites = 50L, species = 11L)
  X = sim$env_weights
  Y = sim$response

  testthat::expect_error(sjSDM(X = data.frame(X), Y), NA)
  testthat::expect_error(sjSDM(X = data.frame(X), data.frame(Y)))

  testthat::expect_error(sjSDM(X, Y, iter = 5), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 2L), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L), NA)

  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, df = 7L), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, df = 50L), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, df = 100L), NA)

  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, l1_coefs = 0.1), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, l2_coefs = 0.1), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, l1_cov  = 0.1), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, l2_cov = 0.1), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, l2_cov = 0.1, l1_cov = 0.1, l1_coefs = 0.1, l2_coefs = 0.1), NA)

  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, learning_rate = 1.0), NA)

  testthat::expect_error({model = sjSDM(X, Y, iter = 1, step_size = 50L, formula = ~X1:X2 + X3)}, NA)
  # testthat::expect_equal(length(model$model$weights_r[[1]]), 2L)
  testthat::expect_error({model = sjSDM(X, Y, iter = 1, step_size = 50L, formula = ~ 0 + X1:X2 + X3)}, NA)



  sim = simulate_SDM(sites = 50L, species = 7L)
  X = sim$env_weights
  Y = sim$response

  testthat::expect_error(sjSDM(X = data.frame(X), Y)<NA)
  testthat::expect_error(sjSDM(X = data.frame(X), data.frame(Y)))

  testthat::expect_error(sjSDM(X, Y, iter = 5), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 2L), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L), NA)

  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, df = 7L), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, df = 50L), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, df = 100L), NA)

  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, l1_coefs = 0.1), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, l2_coefs = 0.1), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, l1_cov  = 0.1), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, l2_cov = 0.1), NA)
  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, l2_cov = 0.1, l1_cov = 0.1, l1_coefs = 0.1, l2_coefs = 0.1), NA)

  testthat::expect_error(sjSDM(X, Y, iter = 1, step_size = 50L, learning_rate = 1.0), NA)

  testthat::expect_error({model = sjSDM(X, Y, iter = 1, step_size = 50L, formula = ~X1:X2 + X3)}, NA)
  # testthat::expect_equal(length(model$model$weights_r[[1]]), 2L)
  testthat::expect_error({model = sjSDM(X, Y, iter = 1, step_size = 50L, formula = ~ 0 + X1:X2 + X3)}, NA)
  # testthat::expect_equal(length(model$model$weights_r[[1]]), 1L)

})



testthat::test_that("sjSDM methods", {
  skip_if_no_torch()

  sim = simulate_SDM(sites = 50L, species = 11L)
  X = sim$env_weights
  Y = sim$response

  testthat::expect_error({model = sjSDM(X, Y, iter = 5, step_size = 50L)}, NA)
  testthat::expect_error(print(model), NA)
  testthat::expect_error(coef(model), NA)
  testthat::expect_error(summary(model), NA)

})
