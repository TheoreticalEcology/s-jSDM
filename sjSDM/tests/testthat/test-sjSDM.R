context("sjSDM")

source("utils.R")

test_model = function(occ = NULL, env, spatial=NULL, biotic = bioticStruct(), 
                      iter = 1L, step_size = 10L, se=FALSE, family = stats::binomial("logit"), context = "") {
    sjSDM:::check_module()
    if(torch$cuda$is_available()) device = "gpu"
    else device = "cpu"
    testthat::expect_error({model = sjSDM(!!occ, env=!!env, 
                                          spatial = !!spatial, 
                                          biotic = !!biotic,
                                          iter = !!iter, 
                                          step_size = !!step_size,
                                          se = !!se,
                                          family=!!family, 
                                          device = device,
                                          sampling = 5L)}, NA)
    testthat::expect_error({.k = testthat::capture_output(print(model))}, NA)
    testthat::expect_error({ .k = testthat::capture_output(coef(model)) }, NA)
    testthat::expect_error({ .k = testthat::capture_output(summary(model)) }, NA)
    testthat::expect_error(logLik(model), NA)
    testthat::expect_error({ .k= testthat::capture_output(predict(model, batch_size=step_size)) }, NA)
    if(inherits(env, "matrix"))testthat::expect_error({ .k= testthat::capture_output(predict(model, newdata = env, batch_size=step_size)) }, NA)
}

# 
# testthat::test_that("sjSDM functionality", {
#   skip_if_no_torch()
  
  library(sjSDM)
  
  test_sims = matrix(c(4:15, 15:4), ncol = 2L)
  testthat::test_that("sjSDM base", {
    skip_if_no_torch()
    for(i in 1:nrow(test_sims)) {
      sim = simulate_SDM(sites = 50L, species = test_sims[i, 1], env = test_sims[i, 2])
      X1 = sim$env_weights
      Y1 = sim$response
      test_model(Y1, X1)
    }
  })
  
  sim = simulate_SDM(sites = 50L, species = 12, env = 3)
  X1 = sim$env_weights
  Y1 = sim$response
  
  
  # iter, batch_size, se, link
  Funcs = list(
    list(5, 2, FALSE, stats::binomial("logit")),
    list(5, 23, FALSE, stats::binomial("probit")),
    list(5, 40, FALSE, stats::poisson("log")),
    list(5, 40, FALSE, stats::gaussian()),
    list(5, 20, TRUE, stats::binomial()),
    list(5, 20, TRUE, stats::poisson()),
    list(5, 20, TRUE, stats::binomial("probit"))
  )
  testthat::test_that("sjSDM Func", {
    skip_if_no_torch()
    for(i in 1:length(Funcs)) {
      test_model(Y1, env = linear(X1), iter = Funcs[[i]][[1]], step_size =  Funcs[[i]][[2]],  se = Funcs[[i]][[3]], family =  Funcs[[i]][[4]])
    }
  })
  
  
  biotic = list(
    bioticStruct(4L),
    bioticStruct(4L, lambda = 0.1),
    bioticStruct(4L, lambda = 0.1, alpha = 0.0),
    bioticStruct(4L, lambda = 0.1, alpha = 1.0),
    bioticStruct(4L, lambda = 0.1, on_diag = TRUE),
    bioticStruct(4L, lambda = 0.1, inverse = TRUE)
  )
  testthat::test_that("sjSDM Biotic", {
    skip_if_no_torch()
    for(i in 1:length(biotic)) {
      test_model(Y1, env=linear(X1), biotic = biotic[[i]])
    }
  })
  
  
  envs = list(
    linear(X1, ~0+.),
    linear(X1, ~1+.),
    linear(X1, ~0+X1:X2),
    linear(X1, ~.),
    linear(X1, lambda = 0.1),
    linear(X1, lambda = 0.1, alpha=0.0),
    linear(X1, lambda = 0.1, alpha=1.0)
  )
  testthat::test_that("sjSDM env", {
    skip_if_no_torch()
    for(i in 1:length(envs)) {
      test_model(Y1, env = envs[[i]])
    }
  })
  
  
  DNN = list(
    DNN(X1, ~0+.),
    DNN(X1, ~1+.),
    DNN(X1, ~0+X1:X2),
    DNN(X1, ~.),
    DNN(X1, lambda = 0.1),
    DNN(X1, lambda = 0.1, alpha=0.0),
    DNN(X1, lambda = 0.1, alpha=1.0),
    DNN(X1, hidden = c(3,3,3),lambda = 0.1, alpha=1.0),
    DNN(X1, hidden = c(4,3,6),lambda = 0.1, alpha=0.0),
    DNN(X1, hidden = c(4,3,6),activation = "relu", lambda = 0.1, alpha=1.0),
    DNN(X1, lambda = 0.1, dropout = 0.3),
    DNN(X1, lambda = 0.1, alpha=0.0 , dropout = 0.1),
    DNN(X1, lambda = 0.1, alpha=1.0, dropout = 0.3),
    DNN(X1, hidden = c(3,3,3),lambda = 0.1, alpha=1.0, dropout = 0.3),
    DNN(X1, hidden = c(4,3,6),lambda = 0.1, alpha=0.0, dropout = 0.3),
    DNN(X1, hidden = c(4,3,6),activation = "relu", lambda = 0.1, alpha=1.0, dropout = 0.3),
    DNN(X1, hidden = c(4,3,6),activation = "tanh", lambda = 0.1, alpha=1.0),
    DNN(X1, hidden = c(4,3,6),activation = "sigmoid", lambda = 0.1, alpha=1.0),
    DNN(X1, hidden = c(4,3,6),activation = c("relu", "tanh", "sigmoid"), lambda = 0.1, alpha=1.0)
  )
  testthat::test_that("sjSDM DNN", {
    skip_if_no_torch()
    for(i in 1:length(DNN)) {
      test_model(Y1, env = DNN[[i]])
    }
  })
  
  SP = matrix(rnorm(100), 50, 2)
  Spatial = list(
    linear(SP, ~0+.),
    linear(SP, ~0+X1:X2),
    linear(SP, ~.),
    linear(SP, lambda = 0.1, alpha=1.0),
    linear(SP, lambda = 0.1, alpha=0.0),
    DNN(SP, hidden = c(3,3,3),lambda = 0.1, alpha=1.0, dropout = 0.3),
    DNN(SP, hidden = c(4,3,6),lambda = 0.1, alpha=0.0)
  )
  testthat::test_that("sjSDM Spatial", {
    skip_if_no_torch()
    for(i in 1:length(Spatial)) {
      test_model(Y1, env = linear(X1), spatial = Spatial[[i]])
    }
  })
  
  
  Spatial = list(
    linear(SP, ~0+.),
    linear(SP, lambda = 0.1, alpha=0.5),
    DNN(SP, hidden = c(3,3,3),lambda = 0.1, alpha=1.0, dropout = 0.3),
    DNN(SP, hidden = c(4,3,6),lambda = 0.1, alpha=0.0)
  )
  
  Env = list(
    linear(X1, ~0+.),
    linear(X1, lambda = 0.1, alpha=0.5),
    DNN(X1, hidden = c(3,3,3),lambda = 0.1, alpha=1.0),
    DNN(X1, hidden = c(4,3,6),lambda = 0.1, alpha=0.0, dropout = 0.3)
  )
  testthat::test_that("sjSDM Mix", {
    skip_if_no_torch()
    for(i in 1:length(Spatial)) {
      test_model(Y1, env = Env[[i]], spatial = Spatial[[i]])
    }
  })
  

testthat::test_that("sjSDM reload model", {
  skip_if_no_torch()
  sjSDM:::check_module()
  if(torch$cuda$is_available()) device = "gpu"
  else device = "cpu"
  
  com = simulate_SDM(env = 3L, species = 5L, sites = 100L)
  model = sjSDM(Y = com$response,env = com$env_weights, iter = 2L, device=device)
  saveRDS(model, "test_model.RDS")
  model = readRDS("test_model.RDS")
  testthat::expect_error(predict(model), NA)
  testthat::expect_error(predict(model, newdata = com$env_weights), NA)
  
  SP = matrix(rnorm(200), 100, 2)
  model = sjSDM(Y = com$response,env = com$env_weights,spatial = linear(SP), iter = 2L, device="cpu")
  saveRDS(model, "test_model.RDS")
  model = readRDS("test_model.RDS")
  testthat::expect_error(predict(model), NA)
  testthat::expect_error(predict(model, newdata = com$env_weights, SP = SP), NA)
  
  com = simulate_SDM(env = 3L, species = 5L, sites = 100L)
  model = sjSDM(Y =com$response,env = DNN(com$env_weights), iter = 2L, device=device)
  saveRDS(model, "test_model.RDS")
  model2 = readRDS("test_model.RDS")
  testthat::expect_error(predict(model2), NA)
  testthat::expect_error(predict(model2, newdata = com$env_weights), NA)
  
  SP = matrix(rnorm(200), 100, 2)
  model = sjSDM(Y = com$response,env = DNN(com$env_weights),spatial = DNN(SP), iter = 2L, device=device)
  saveRDS(model, "test_model.RDS")
  model2 = readRDS("test_model.RDS")
  testthat::expect_error(predict(model2), NA)
  testthat::expect_error(predict(model2, newdata = com$env_weights, SP = SP), NA)
  file.remove("test_model.RDS")
})



