# context("sjSDM")
# 
# source("utils.R")
# 
# testthat::test_that("sjSDM functionality", {
#   skip_if_no_torch()
#   
#   library(sjSDM)
#   Re = rnorm(50,0,0.3)
#   sim = simulate_SDM(sites = 50L, species = 11L,Re = Re)
#   X1 = sim$env_weights
#   Y1 = sim$response
#   testthat::expect_error({model = sjSDM(env = data.frame(sim$env_weights),spatial = spatialRE(1:50), Y = Y1, iter = 5L)}, NA)
#   testthat::expect_error({summary(model)}, NA)
#   testthat::expect_error({print(model)}, NA)
#   testthat::expect_error({logLik(model)}, NA)
#   testthat::expect_error({simulate(model)}, NA)
#   testthat::expect_error({predict(model)}, NA)
#   testthat::expect_error({predict(model, newdata = X1)}, NA)
#   testthat::expect_error({ranef(model)}, NA)
#   
#   RR = sample(1:5, 50, TRUE)
#   Re = rnorm(5, 0, 0.1)[RR]
#   sim = simulate_SDM(sites = 50L, species = 11L,Re = Re)
#   X1 = sim$env_weights
#   Y1 = sim$response
#   testthat::expect_error({model = sjSDM(env = data.frame(sim$env_weights),spatial = spatialRE(as.factor(RR)), Y = Y1, iter = 5L)}, NA)
#   testthat::expect_error({summary(model)}, NA)
#   testthat::expect_error({print(model)}, NA)
#   testthat::expect_error({logLik(model)}, NA)
#   testthat::expect_error({simulate(model)}, NA)
#   testthat::expect_error({predict(model)}, NA)
#   testthat::expect_error({predict(model, newdata = X1)}, NA)
#   testthat::expect_error({ranef(model)}, NA)
# })