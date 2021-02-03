context("sLVM")

source("utils.R")

test_model = function(Y = NULL, X = NULL, formula = as.formula("~0+."), lv=2L,
                      family=binomial("probit"), priors=list(3, 1, 1),
                      posterior = "DiagonalNormal",
                      iter = 1L,
                      step_size = 20L, lr = list(0.1)) {
  
    sjSDM:::check_module()
    if(torch$cuda$is_available()) device = "gpu"
    else device = "cpu"
    testthat::expect_error({model <<- sLVM(Y=!!Y, X=!!X, formula=!!formula,
                                         family = !!family, priors=!!priors,
                                         posterior = !!posterior,iter=!!iter,
                                         step_size = !!step_size, lr=!!lr,
                                         device = device)}, NA)
    testthat::expect_error({.k = testthat::capture_output(print(model))}, NA)
    testthat::expect_error({ .k = testthat::capture_output(coef(model)) }, NA)
    testthat::expect_error({ .k = testthat::capture_output(summary(model)) }, NA)
    testthat::expect_error(logLik(model), NA)
    testthat::expect_error({ .k= testthat::capture_output(predict(model, batch_size=step_size)) }, NA)
    testthat::expect_error({ .k= testthat::capture_output(predict(model, newdata=X)) }, NA)
    testthat::expect_error({ .k= testthat::capture_output(predict(model, newdata=X, mean_field = FALSE)) }, NA)
}

# 
# testthat::test_that("sjSDM functionality", {
#   skip_if_no_torch()
  
  library(sjSDM)
  
  test_sims = matrix(c(4:15, 15:4), ncol = 2L)
  testthat::test_that("sLVM base", {
    testthat::skip_on_cran()
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
  
  
  # iter, batch_size, family, link
  Funcs = list(
    list(2, binomial("probit")),
    list(2, binomial("logit")),
    list(2, poisson("log")),
    #list(2, poisson("identity")),
    list(23, binomial("probit")),
    list(23, binomial("logit")),
    list(23, poisson("log"))
    #list(23, poisson("identity"))

  )
  testthat::test_that("sLVM Func", {
    testthat::skip_on_cran()
    skip_if_no_torch()
    for(i in 1:length(Funcs)) {
      test_model(Y1, X1, step_size =  Funcs[[i]][[1]],  family=Funcs[[i]][[2]])
    }
  })
  
  
  
  envs = list(
    as.formula(as.character("~0+.")),
    as.formula(as.character("~1+.")),
    as.formula(as.character("~0+X1:X2")),
    as.formula(as.character("~."))
  )
  testthat::test_that("sLVM env", {
    skip_if_no_torch()
    for(i in 1:length(envs)) {
      test_model(Y1, X1, formula = envs[[i]])
    }
  })
  
  guide_df_prior = list(
    list("DiagonalNormal", list(2.0, 3.0, 3.0), 2L),
    list("DiagonalNormal", list(2.0, 3.0, 3.0), 4L),
    list("DiagonalNormal", list(2.0, 3.0, 3.0), 7L),
    list("LaplaceApproximation", list(1.0, 3.0, 3.0), 2L),
    list("LowRankMultivariateNormal", list(0.3, 0.3, 0.3), 2L),
    list("Delta", list(2.0, 3.0, 3.0), 2L)
  )
  testthat::test_that("sLVM env", {
    testthat::skip_on_cran()
    skip_if_no_torch()
    for(i in 1:length(guide_df_prior)) {
      test_model(Y1, X1, posterior=guide_df_prior[[i]][[1]], priors = guide_df_prior[[i]][[2]], lv = guide_df_prior[[i]][[3]])
    }
  })
  

testthat::test_that("sLVM reload model", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  com = simulate_SDM(env = 3L, species = 5L, sites = 100L)
  model = sLVM(Y = com$response,X = com$env_weights, iter = 2L, family = binomial("probit"))
  saveRDS(model, "test_model.RDS")
  model = readRDS("test_model.RDS")
  testthat::expect_error(predict(model), NA)
  testthat::expect_error(predict(model, newdata = com$env_weights), NA)
  file.remove("test_model.RDS")
})



