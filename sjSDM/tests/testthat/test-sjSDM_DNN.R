context("sjSDM_model")

source("utils.R")

testthat::test_that("sjSDM functionality", {
  skip_if_no_torch()
  
  sim = simulate_SDM(sites = 50L, species = 11L)
  X = sim$env_weights
  Y = sim$response
  
  testthat::expect_error(sjSDM_DNN(X, Y, iter = 1, step_size = 50L, l2_cov = 0.1), NA)
  testthat::expect_error(sjSDM_DNN(X, Y, iter = 1, step_size = 50L, l2_cov = 0.1, l1_cov = 0.1, l1_coefs = 0.1, l2_coefs = 0.1), NA)
  
  testthat::expect_error(sjSDM_DNN(X, Y, iter = 1, step_size = 50L, learning_rate = 1.0), NA)
  
  testthat::expect_error({model = sjSDM_DNN(X, Y, iter = 1, step_size = 50L, formula = ~X1:X2 + X3)}, NA)
  testthat::expect_error({model = sjSDM_DNN(X, Y, iter = 1, step_size = 50L, formula = ~ 0 + X1:X2 + X3)}, NA)
  
  testthat::expect_error(summary(model), NA)
  testthat::expect_error(getCov(model), NA)
  testthat::expect_error(predict(model), NA)
  testthat::expect_error(predict(model, newdata = X), NA)
  testthat::expect_error(plot(model), NA)
  testthat::expect_error({w = getWeights(model)}, NA)
  testthat::expect_error(setWeights(model), NA)
  testthat::expect_error(setWeights(model, w), NA)
  testthat::expect_error({
    w = getWeights(model)
    w$layers[[1]][[1]][,] = 0
    setWeights(model, w)
    w = getWeights(model)
    stopifnot( !any(!as.vector(w$layers[[1]][[1]][,] == 0)))
    }, NA)
  
  
  sim = simulate_SDM(sites = 100L, species = 51L, env = 10L)
  X = sim$env_weights
  Y = sim$response
  
  testthat::expect_error(sjSDM_DNN(X, Y, iter = 1, step_size = 50L, l2_cov = 0.1), NA)
  testthat::expect_error(sjSDM_DNN(X, Y, iter = 1, step_size = 50L, l2_cov = 0.1, l1_cov = 0.1, l1_coefs = 0.1, l2_coefs = 0.1), NA)
  
  testthat::expect_error(sjSDM_DNN(X, Y, iter = 1, step_size = 50L, learning_rate = 1.0), NA)
  
  testthat::expect_error({model = sjSDM_DNN(X, Y, iter = 1, step_size = 50L, formula = ~X1:X2 + X3)}, NA)
  testthat::expect_error({model = sjSDM_DNN(X, Y, iter = 1, step_size = 50L, formula = ~ 0 + X1:X2 + X3, df = 30L, hidden = c(5L, 5L, 5L, 10L, 10L))}, NA)
  
  testthat::expect_error(summary(model), NA)
  testthat::expect_error(getCov(model), NA)
  testthat::expect_error(predict(model), NA)
  testthat::expect_error(predict(model, newdata = X), NA)
  testthat::expect_error(plot(model), NA)
  testthat::expect_error({w = getWeights(model)}, NA)
  testthat::expect_error(setWeights(model), NA)
  testthat::expect_error(setWeights(model, w), NA)
  testthat::expect_error({
    w = getWeights(model)
    w$layers[[1]][[1]][,] = 0
    setWeights(model, w)
    w = getWeights(model)
    stopifnot( !any(!as.vector(w$layers[[1]][[1]][,] == 0)))
  }, NA)
  
  
  
})