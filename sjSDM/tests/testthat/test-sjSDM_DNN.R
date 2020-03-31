context("sjSDM_model")

source("utils.R")

testthat::test_that("sjSDM functionality", {
  skip_if_no_torch()
  
  sim = simulate_SDM(sites = 50L, species = 11L)
  X1 = sim$env_weights
  Y1 = sim$response
  
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1, formula = ~X1:X2 + X3),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1, formula = ~0+X1:X2 + X3),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1, lambda = 0.01),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1, lambda = 0.01, alpha = 0.0),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1, lambda = 0.01, alpha = 1.0),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1, lambda = 0.01, alpha = 0.5),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1, lambda = 0.01, alpha = 0.5, hidden = c(3L, 3L, 3L)),iter = 1, step_size = 50L), NA)
  testthat::expect_error({model = sjSDM(Y1, env = envDNN(X1, lambda = 0.01, alpha = 0.5, hidden = c(3L, 3L, 3L), activation = c("tanh", "relu", "tanh")),iter = 1, step_size = 50L)}, NA)
  
  testthat::expect_error(summary(model), NA)
  testthat::expect_error(getCov(model), NA)
  testthat::expect_error(predict(model), NA)
  testthat::expect_error(predict(model, newdata = X1), NA)
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
  X1 = sim$env_weights
  Y1 = sim$response
  
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1, formula = ~X1:X2 + X3),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1, formula = ~0+X1:X2 + X3),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1, lambda = 0.01),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1, lambda = 0.01, alpha = 0.0),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1, lambda = 0.01, alpha = 1.0),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1, lambda = 0.01, alpha = 0.5),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envDNN(X1, lambda = 0.01, alpha = 0.5, hidden = c(3L, 3L, 3L)),iter = 1, step_size = 50L), NA)
  testthat::expect_error({model = sjSDM(Y1, env = envDNN(X1, lambda = 0.01, alpha = 0.5, hidden = c(3L, 3L, 3L), activation = c("tanh", "relu", "tanh")),iter = 1, step_size = 50L)}, NA)
  
  testthat::expect_error(summary(model), NA)
  testthat::expect_error(getCov(model), NA)
  testthat::expect_error(predict(model), NA)
  testthat::expect_error(predict(model, newdata = X1), NA)
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