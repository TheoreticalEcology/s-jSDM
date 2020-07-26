context("sjSDM")

source("utils.R")

test_model = function(occ = NULL, env, spatial=NULL, biotic = bioticStruct(), 
                      iter = 1L, step_size = 10L, se=FALSE, link = "logit", context = "") {
  
    sjSDM:::check_module()
    if(torch$cuda$is_available()) device = "gpu"
    else device = "cpu"
    testthat::expect_error({model = sjSDM(!!occ, env=!!env, 
                                          spatial = !!spatial, 
                                          biotic = !!biotic,
                                          iter = !!iter, 
                                          step_size = !!step_size,
                                          se = !!se,
                                          link = !!link,
                                          device = device)}, NA)
    testthat::expect_error({res = anova(model, cv = 2L)}, NA)
    testthat::expect_error({plot(res)}, NA)
}

# 
# testthat::test_that("sjSDM functionality", {
#   skip_if_no_torch()
  
  library(sjSDM)
  
  sim = simulate_SDM(sites = 50L, species = 12, env = 3)
  X1 = sim$env_weights
  Y1 = sim$response
  
  
  # iter, batch_size, se, link
  Funcs = list(
    list(5, 2, FALSE, "logit"),
    list(5, 23, FALSE, "logit"),
    list(5, 40, FALSE, "probit")
  )
  testthat::test_that("sjSDM anova Func", {
    skip_if_no_torch()
    for(i in 1:length(Funcs)) {
      test_model(Y1, env = linear(X1), iter = Funcs[[i]][[1]], step_size =  Funcs[[i]][[2]],  se = Funcs[[i]][[3]], link =  Funcs[[i]][[4]])
    }
  })
  
  
  biotic = list(
    bioticStruct(4L),
    bioticStruct(4L, lambda = 0.1, alpha = 1.0)
  )
  testthat::test_that("sjSDM anova Biotic", {
    skip_if_no_torch()
    for(i in 1:length(biotic)) {
      test_model(Y1, env=linear(X1), biotic = biotic[[i]])
    }
  })
  
  
  envs = list(
    linear(X1, ~0+X1:X2),
    linear(X1, lambda = 0.1)
  )
  testthat::test_that("sjSDM anova env", {
    skip_if_no_torch()
    for(i in 1:length(envs)) {
      test_model(Y1, env = envs[[i]])
    }
  })
  
  
  DNN = list(
    DNN(X1, hidden = c(3,3,3),lambda = 0.1, alpha=1.0)
  )
  testthat::test_that("sjSDM anova DNN", {
    skip_if_no_torch()
    for(i in 1:length(DNN)) {
      test_model(Y1, env = DNN[[i]])
    }
  })
  
  SP = matrix(rnorm(100), 50, 2)
  Spatial = list(
    linear(SP, ~0+X1:X2),
    linear(SP, lambda = 0.1, alpha=0.0),
    DNN(SP, hidden = c(4,3,6),lambda = 0.1, alpha=0.0)
  )
  testthat::test_that("sjSDM anova Spatial", {
    skip_if_no_torch()
    for(i in 1:length(Spatial)) {
      test_model(Y1, env = linear(X1), spatial = Spatial[[i]])
    }
  })
  
  
  Spatial = list(
    linear(SP, ~0+.),
    DNN(SP, hidden = c(4,3,6),lambda = 0.1, alpha=0.0)
  )
  
  Env = list(
    linear(X1, lambda = 0.1, alpha=0.5),
    DNN(X1, hidden = c(4,3,6),lambda = 0.1, alpha=0.0)
  )
  testthat::test_that("sjSDM anova Mix", {
    skip_if_no_torch()
    for(i in 1:length(Spatial)) {
      test_model(Y1, env = Env[[i]], spatial = Spatial[[i]])
    }
  })
  


