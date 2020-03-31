context("sjSDM")

source("utils.R")

testthat::test_that("sjSDM functionality", {
  skip_if_no_torch()
  
  library(sjSDM)

  sim = simulate_SDM(sites = 50L, species = 11L)
  X1 = sim$env_weights
  Y1 = sim$response
  

  testthat::expect_error(sjSDM(env = data.frame(sim$env_weights), Y = Y1), NA)
  testthat::expect_error(sjSDM(env = data.frame(sim$env_weights), Y = data.frame(Y1)))
  testthat::expect_error(sjSDM(Y1, X1, iter = 5), NA)
  testthat::expect_error(sjSDM(Y1, X1, iter = 1, step_size = 2L), NA)
  testthat::expect_error(sjSDM(Y1, X1, iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, X1, iter = 1, step_size = 50L, biotic = bioticStruct(df = 7L)), NA)
  testthat::expect_error(sjSDM(Y1, X1, iter = 1, step_size = 50L, biotic = bioticStruct(df = 50L)), NA)
  testthat::expect_error(sjSDM(Y1, X1, iter = 1, step_size = 50L, biotic = bioticStruct(df = 100L)), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, lambda = 0.1), iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, lambda = 0.1, alpha = 0.0), iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, lambda = 0.1, alpha = 1.0), iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, lambda = 0.1), biotic = bioticStruct(20L, lambda = 0.1, alpha = 0.0),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, lambda = 0.1), biotic = bioticStruct(20L, lambda = 0.1, alpha = 0.5),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, lambda = 0.1), biotic = bioticStruct(20L, lambda = 0.1, alpha = 1.0),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, lambda = 0.1), biotic = bioticStruct(20L, lambda = 0.1, alpha = 0.5, on_diag = TRUE),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, formula = ~X1:X2 + X3, lambda = 0.1), iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, formula = ~0 + X1:X2), iter = 1, step_size = 50L), NA)


  sim = simulate_SDM(sites = 50L, species = 7L)
  X1 = sim$env_weights
  Y1 = sim$response

  testthat::expect_error(sjSDM(env = (X1), Y = Y1), NA)
  testthat::expect_error(sjSDM(env = (X1), Y = data.frame(Y1)))
  testthat::expect_error(sjSDM(Y1, X1, iter = 5), NA)
  testthat::expect_error(sjSDM(Y1, X1, iter = 1, step_size = 2L), NA)
  testthat::expect_error(sjSDM(Y1, X1, iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, X1, iter = 1, step_size = 50L, biotic = bioticStruct(df = 7L)), NA)
  testthat::expect_error(sjSDM(Y1, X1, iter = 1, step_size = 50L, biotic = bioticStruct(df = 50L)), NA)
  testthat::expect_error(sjSDM(Y1, X1, iter = 1, step_size = 50L, biotic = bioticStruct(df = 100L)), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, lambda = 0.1), iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, lambda = 0.1, alpha = 0.0), iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, lambda = 0.1, alpha = 1.0), iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, lambda = 0.1), biotic = bioticStruct(20L, lambda = 0.1, alpha = 0.0),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, lambda = 0.1), biotic = bioticStruct(20L, lambda = 0.1, alpha = 0.5),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, lambda = 0.1), biotic = bioticStruct(20L, lambda = 0.1, alpha = 1.0),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, lambda = 0.1), biotic = bioticStruct(20L, lambda = 0.1, alpha = 0.5, on_diag = TRUE),iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, formula = ~X1:X2 + X3, lambda = 0.1), iter = 1, step_size = 50L), NA)
  testthat::expect_error(sjSDM(Y1, env = envLinear(X1, formula = ~0 + X1:X2), iter = 1, step_size = 50L), NA)

})



testthat::test_that("sjSDM methods", {
  skip_if_no_torch()

  sim = simulate_SDM(sites = 50L, species = 11L)
  X1 = sim$env_weights
  Y1 = sim$response

  testthat::expect_error({model = sjSDM(Y1, X1, iter = 5, step_size = 50L)}, NA)
  testthat::expect_error(print(model), NA)
  testthat::expect_error(coef(model), NA)
  testthat::expect_error(summary(model), NA)
  
  testthat::expect_error({model = sjSDM(Y1, X1, iter = 5, step_size = 50L,se=TRUE)}, NA)
  testthat::expect_error(summary(model), NA)
  testthat::expect_error(getSe(model), NA)
  
  testthat::expect_error({model = sjSDM(Y1, X1, iter = 5, step_size = 50L,se=FALSE)}, NA)
  testthat::expect_error({model=getSe(model)}, NA)
  testthat::expect_error({summary(model)}, NA)
  
  colnames(Y1) = 1:11
  testthat::expect_error({model = sjSDM(Y1, X1, iter = 5, step_size = 50L,se=FALSE)}, NA)
  testthat::expect_error({model=getSe(model, step_size = 30)}, NA)
  testthat::expect_error({ sum = summary(model)}, NA)
  
  testthat::expect_error({logLik(model)}, NA)
  testthat::expect_error({simulate(model)}, NA)
  testthat::expect_error({simulate(model, nsim = 10)}, NA)

})
