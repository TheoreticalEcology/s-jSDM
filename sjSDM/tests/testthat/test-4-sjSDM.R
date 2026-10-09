source("utils.R")

test_model = function(occ = NULL, env, spatial=NULL, biotic = bioticStruct(), 
                      iter = 1L, step_size = 10L, se=FALSE, 
                      family = stats::binomial("logit"), 
                      control = sjSDMControl(),
                      context = "") {
    sjSDM:::check_module()
    device = is_gpu_available()
    testthat::expect_error({model = sjSDM(!!occ, env=!!env, 
                                          spatial = !!spatial, 
                                          biotic = !!biotic,
                                          iter = !!iter, 
                                          step_size = !!step_size,
                                          se = !!se,
                                          family=!!family, 
                                          control = control,
                                          device = device,
                                          sampling = 5L, verbose = FALSE)}, NA)
    testthat::expect_error({.k = testthat::capture_output(print(model))}, NA)
    testthat::expect_error({ .k = testthat::capture_output(coef(model)) }, NA)
    testthat::expect_error({.k = testthat::capture_output(print(model))}, NA)
    testthat::expect_error({ .k = testthat::capture_output(summary(model)) }, NA)
    testthat::expect_error({ .k = testthat::capture_output(Rsquared(model)) }, NA)
    testthat::expect_error({ .k = testthat::capture_output(update(model, env_formula=~1)) }, NA)
    if(inherits(model, "spatial")) testthat::expect_error({ .k = testthat::capture_output(update(model, env_formula=~1, spatial_formula =~0)) }, NA)
    testthat::expect_false(any(is.na(model$history)))
    testthat::expect_error({ .k= testthat::capture_output(predict(model, batch_size=step_size)) }, NA)
    if(inherits(env, "matrix"))testthat::expect_error({ .k= testthat::capture_output(predict(model, newdata = env, batch_size=step_size)) }, NA)
}

# 
# testthat::test_that("sjSDM functionality", {
#   skip_if_no_torch()
  
  library(sjSDM)
  
  test_sims = matrix(c(4:15, 15:4), ncol = 2L)
  testthat::test_that("sjSDM base", {
    testthat::skip_on_cran()
    testthat::skip_on_ci()
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
    list(5, 20, TRUE, stats::binomial("probit")),
    list(5, 20, TRUE, "nbinom")
  )
  testthat::test_that("sjSDM Func", {
    testthat::skip_on_cran()
    testthat::skip_on_ci()
    skip_if_no_torch()
    for(i in 1:length(Funcs)) {
      test_model(Y1, env = linear(X1), iter = Funcs[[i]][[1]], step_size =  Funcs[[i]][[2]],  se = Funcs[[i]][[3]], family =  Funcs[[i]][[4]])
    }
  })
  
  
  # sjSDM controls
  controls = list(
    sjSDMControl(optimizer = RMSprop()),
    sjSDMControl(optimizer = Adamax()),
    sjSDMControl(optimizer = AdaBound()),
    sjSDMControl(optimizer = AccSGD()),
    sjSDMControl(optimizer = SGD()),
    sjSDMControl(optimizer = madgrad()),
    sjSDMControl(optimizer = RMSprop(), scheduler = 2, lr_reduce_factor = 0.1),
    sjSDMControl(optimizer = RMSprop(), scheduler = 2, lr_reduce_factor = 0.99),
    sjSDMControl(optimizer = RMSprop(), early_stopping_training = 2L)
  )
  testthat::test_that("sjSDM Control", {
    testthat::skip_on_cran()
    testthat::skip_on_ci()
    skip_if_no_torch()
    for(i in 1:length(controls)) {
      test_model(Y1, env = linear(X1), iter = 20L, control = controls[[i]])
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
    testthat::skip_on_cran()
    testthat::skip_on_ci()
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
    testthat::skip_on_cran()
    testthat::skip_on_ci()
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
    DNN(X1, hidden = c(3,3,3),lambda = 0.1, bias = FALSE, alpha=1.0, dropout = 0.3),
    DNN(X1, hidden = c(3,3,3),lambda = 0.1, bias = c(TRUE, FALSE, TRUE), alpha=1.0, dropout = 0.3),
    DNN(X1, hidden = c(3,3,3),lambda = 0.1, bias = c(TRUE, FALSE, TRUE, FALSE), alpha=1.0, dropout = 0.3),
    DNN(X1, hidden = c(4,3,6),lambda = 0.1, alpha=0.0, dropout = 0.3),
    DNN(X1, hidden = c(4,3,6),activation = "relu", lambda = 0.1, alpha=1.0, dropout = 0.3),
    DNN(X1, hidden = c(4,3,6),activation = "selu", lambda = 0.1, alpha=1.0, dropout = 0.3),
    DNN(X1, hidden = c(4,3,6),activation = "leakyrelu", lambda = 0.1, alpha=1.0, dropout = 0.3),
    DNN(X1, hidden = c(4,3,6),activation = "tanh", lambda = 0.1, alpha=1.0),
    DNN(X1, hidden = c(4,3,6),activation = "sigmoid", lambda = 0.1, alpha=1.0),
    DNN(X1, hidden = c(4,3,6),activation = c("relu", "tanh", "sigmoid"), lambda = 0.1, alpha=1.0)
  )
  testthat::test_that("sjSDM DNN", {
    testthat::skip_on_cran()
    testthat::skip_on_ci()
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
    testthat::skip_on_cran()
    testthat::skip_on_ci()
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
    testthat::skip_on_cran()
    testthat::skip_on_ci()
    skip_if_no_torch()
    for(i in 1:length(Spatial)) {
      test_model(Y1, env = Env[[i]], spatial = Spatial[[i]])
    }
  })
  

testthat::test_that("sjSDM reload model", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  sjSDM:::check_module()
  device = is_gpu_available()
  
  com = simulate_SDM(env = 3L, species = 5L, sites = 100L)
  model = sjSDM(Y = com$response,env = com$env_weights, iter = 2L, device=device, verbose = FALSE)
  saveRDS(model, "test_model.RDS")
  model = readRDS("test_model.RDS")
  testthat::expect_error(predict(model), NA)
  testthat::expect_error(predict(model, newdata = com$env_weights), NA)
  
  SP = matrix(rnorm(200), 100, 2)
  model = sjSDM(Y = com$response,env = com$env_weights,spatial = linear(SP), iter = 2L, device="cpu", verbose = FALSE)
  saveRDS(model, "test_model.RDS")
  model = readRDS("test_model.RDS")
  testthat::expect_error(predict(model), NA)
  testthat::expect_error(predict(model, newdata = com$env_weights, SP = SP), NA)
  
  com = simulate_SDM(env = 3L, species = 5L, sites = 100L)
  model = sjSDM(Y =com$response,env = DNN(com$env_weights), iter = 2L, device=device, verbose = FALSE)
  saveRDS(model, "test_model.RDS")
  model2 = readRDS("test_model.RDS")
  testthat::expect_error(predict(model2), NA)
  testthat::expect_error(predict(model2, newdata = com$env_weights), NA)
  
  SP = matrix(rnorm(200), 100, 2)
  model = sjSDM(Y = com$response,env = DNN(com$env_weights, bias = TRUE),spatial = DNN(SP), iter = 2L, device=device, verbose = FALSE)
  saveRDS(model, "test_model.RDS")
  model2 = readRDS("test_model.RDS")
  testthat::expect_error(predict(model2), NA)
  testthat::expect_error(predict(model2, newdata = com$env_weights, SP = SP), NA)
  file.remove("test_model.RDS")
  
  
  SP = matrix(rnorm(200), 100, 2)
  model = sjSDM(Y = com$response,env = DNN(com$env_weights, bias = c(TRUE, FALSE, TRUE, FALSE)),spatial = DNN(SP), iter = 2L, device=device, verbose = FALSE)
  saveRDS(model, "test_model.RDS")
  model2 = readRDS("test_model.RDS")
  testthat::expect_error(predict(model2), NA)
  testthat::expect_error(predict(model2, newdata = com$env_weights, SP = SP), NA)
  model2 = sjSDM:::checkModel(model2)
  dims1 = lapply(sjSDM:::force_r(model2$model$env_weights), dim)
  dims2 = lapply(sjSDM:::force_r(model$model$env_weights ), dim)
  testthat::expect_equal(dims1, dims2, info = "dimensions of saved model does not match")
  file.remove("test_model.RDS")
})


testthat::test_that("marginal predictions integrate the latent factor out", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  # predict() used to return link(mu), the value at latent z = 0, which is not the marginal
  # whenever the associations are non-zero. The marginal is E_z[link(mu + z sigma')]; for
  # probit that is pnorm(mu / sqrt(diag(sigma sigma' + I))) and for a log link
  # exp(mu + rowSums(sigma^2)/2).
  set.seed(42)
  n = 60L; sp = 4L; df = 2L
  X = matrix(rnorm(n * 2), n, 2)
  sig = matrix(c(rep(0.8, sp), rep(-0.5, sp)), sp, df)
  guard = function(p) p * 0.999999 + 0.0000005

  Y = matrix(rbinom(n * sp, 1, 0.4), n, sp)
  m = sjSDM(Y = Y, env = linear(X, ~.), biotic = bioticStruct(df = df), iter = 5L,
            sampling = 50L, verbose = FALSE, seed = 7L, family = stats::binomial("probit"))
  sjSDM:::set_state(m$model$loss, list(sigma = sig))
  raw = predict(m, newdata = X, type = "raw")
  sd_k = sqrt(diag(getCov(m)))
  testthat::expect_equal(as.numeric(predict(m, newdata = X)),
                         as.numeric(guard(stats::pnorm(sweep(raw, 2, sd_k, "/")))),
                         tolerance = 1e-5)
  # the old behaviour is still reachable, and is link(mu)
  testthat::expect_equal(as.numeric(predict(m, newdata = X, marginal = FALSE)),
                         as.numeric(guard(stats::pnorm(raw))), tolerance = 1e-5)
  # the two differ, otherwise this test would pass on the broken code
  testthat::expect_gt(max(abs(predict(m, newdata = X) -
                                predict(m, newdata = X, marginal = FALSE))), 0.02)
  # closed form, so repeated calls agree exactly
  testthat::expect_equal(predict(m, newdata = X), predict(m, newdata = X))

  Yp = matrix(rpois(n * sp, 2), n, sp)
  mp = sjSDM(Y = Yp, env = linear(X, ~.), biotic = bioticStruct(df = df), iter = 5L,
             sampling = 50L, verbose = FALSE, seed = 7L, family = stats::poisson())
  sjSDM:::set_state(mp$model$loss, list(sigma = sig))
  rawp = predict(mp, newdata = X, type = "raw")
  testthat::expect_equal(as.numeric(predict(mp, newdata = X)),
                         as.numeric(exp(sweep(rawp, 2, rowSums(sig^2) / 2, "+"))),
                         tolerance = 1e-5)

  Yg = matrix(rnorm(n * sp), n, sp)
  mg = sjSDM(Y = Yg, env = linear(X, ~.), biotic = bioticStruct(df = df), iter = 5L,
             sampling = 50L, verbose = FALSE, seed = 7L, family = stats::gaussian())
  sjSDM:::set_state(mg$model$loss, list(sigma = sig))
  # E[mu + z sigma'] = mu, so the gaussian marginal must be untouched
  testthat::expect_equal(predict(mg, newdata = X),
                         predict(mg, newdata = X, marginal = FALSE), tolerance = 1e-6)

  # with no associations there is nothing to integrate out and the two must coincide
  sjSDM:::set_state(m$model$loss, list(sigma = matrix(0, sp, df)))
  testthat::expect_equal(predict(m, newdata = X),
                         predict(m, newdata = X, marginal = FALSE), tolerance = 1e-5)
})

testthat::test_that("conditional predictions work with only two species", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  # sjSDM 1.0.7 died here with "IndexError: tuple index out of range": conditioning on a
  # single species made Y[, focal], predictions[, focal] and sigma[focal, ] drop to
  # vectors, so the backend read sigma.shape[1] on a 1-d tensor. Three or more species hid
  # it, because then the conditioning set is never a single column.
  set.seed(42)
  n = 80L
  X = matrix(rnorm(n * 2), n, 2)
  fams = list(stats::binomial("probit"), stats::binomial("logit"), stats::poisson(),
              stats::gaussian(), "nbinom")
  for (fam in fams) {
    Y = if (identical(fam, "nbinom")) matrix(rnbinom(n * 2, mu = 2, size = 2), n, 2)
        else if (is.character(fam)) matrix(rbinom(n * 2, 1, 0.4), n, 2)
        else switch(fam$family,
                    poisson  = matrix(rpois(n * 2, 2), n, 2),
                    gaussian = matrix(rnorm(n * 2), n, 2),
                    matrix(rbinom(n * 2, 1, 0.4), n, 2))
    m = sjSDM(Y = Y, env = linear(X, ~.), iter = 5L, sampling = 50L, verbose = FALSE,
              seed = 7L, family = fam)
    for (napos in 1:2) {
      Yc = Y; Yc[, napos] = NA
      p = suppressWarnings(predict(m, newdata = X, Y = Yc, sampling = 200L))
      testthat::expect_equal(nrow(as.matrix(p)), n)
      testthat::expect_equal(ncol(as.matrix(p)), 1L)
      testthat::expect_false(anyNA(p))
    }
  }
})

testthat::test_that("conditioning on a species uses the association matrix", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  # A crash is the obvious failure; silently ignoring the conditioning is the dangerous
  # one. sigma is injected so both cases have a known answer: with sigma = 0 the species
  # are independent and conditioning must change nothing, and with both species loading on
  # one latent they are positively associated and it must.
  set.seed(11)
  n = 150L
  X = matrix(rnorm(n * 2), n, 2)
  Y = matrix(rbinom(n * 2, 1, 0.4), n, 2)
  m = sjSDM(Y = Y, env = linear(X, ~.), biotic = bioticStruct(df = 2L), iter = 5L,
            sampling = 50L, verbose = FALSE, seed = 3L)
  cond = function(y2) {
    Yc = Y; Yc[, 1] = NA; Yc[, 2] = y2
    as.numeric(predict(m, newdata = X, Y = Yc, sampling = 10000L))
  }

  sjSDM:::set_state(m$model$loss, list(sigma = matrix(0, 2, 2)))
  indep = max(abs(cond(1) - cond(0)))
  testthat::expect_lt(indep, 0.02)

  sjSDM:::set_state(m$model$loss, list(sigma = matrix(c(1.5, 1.5, 0, 0), 2, 2)))
  assoc = max(abs(cond(1) - cond(0)))
  testthat::expect_gt(assoc, 0.05)
  testthat::expect_gt(assoc, indep)
})

testthat::test_that("Changing weights", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  sjSDM:::check_module()
  XY = matrix(rnorm(100*2), 100L, 2L)
  com = simulate_SDM(env = 3L, species = 5L, sites = 100L)
  model = sjSDM(Y = com$response,env = com$env_weights, spatial = linear(XY), iter = 2L, verbose = FALSE)
  
  setWeights(model, list(matrix(1.0, 5, 4)))
  testthat::expect_equal(mean(model$model$env_weights[[1]]), 1)
  testthat::expect_equal(dim(model$model$env_weights[[1]]), c(5, 4))
  setWeights(model, list(matrix(2.0, 5, 4)))
  testthat::expect_equal(mean(model$model$env_weights[[1]]), 2)
  testthat::expect_equal(dim(model$model$env_weights[[1]]), c(5, 4))
  
  setWeights(model, list(matrix(3.0, 5, 4), matrix(4.0, 5, 3)))
  testthat::expect_equal(mean(model$model$env_weights[[1]]), 3)
  testthat::expect_equal(mean(model$model$spatial_weights[[1]]), 4)
  testthat::expect_equal(dim(model$model$env_weights[[1]]), c(5, 4))
  testthat::expect_equal(dim(model$model$spatial_weights[[1]]), c(5, 3))
  
  setWeights(model, list(matrix(5.0, 5, 4), list(matrix(6.0, 5, 3))))
  testthat::expect_equal(mean(model$model$env_weights[[1]]), 5)
  testthat::expect_equal(mean(model$model$spatial_weights[[1]]), 6)
  testthat::expect_equal(dim(model$model$env_weights[[1]]), c(5, 4))
  testthat::expect_equal(dim(model$model$spatial_weights[[1]]), c(5, 3))
  
  setWeights(model, list(matrix(5.0, 5, 4), list(matrix(6.0, 5, 3)), matrix(1.0, 5, 5)))
  testthat::expect_equal(mean(model$model$env_weights[[1]]), 5)
  testthat::expect_equal(mean(model$model$spatial_weights[[1]]), 6)
  testthat::expect_equal(mean(model$model$get_sigma), 1)
  testthat::expect_equal(dim(model$model$env_weights[[1]]), c(5, 4))
  testthat::expect_equal(dim(model$model$spatial_weights[[1]]), c(5, 3))
  testthat::expect_equal(dim(model$model$get_sigma), c(5, 5))
  
  setWeights(model, list(matrix(5.0, 5, 4), NULL, matrix(1.0, 5, 5)))
  testthat::expect_equal(mean(model$model$env_weights[[1]]), 5)
  testthat::expect_equal(mean(model$model$spatial_weights[[1]]), 6)
  testthat::expect_equal(mean(model$model$get_sigma), 1)
  testthat::expect_equal(dim(model$model$env_weights[[1]]), c(5, 4))
  testthat::expect_equal(dim(model$model$spatial_weights[[1]]), c(5, 3))
  testthat::expect_equal(dim(model$model$get_sigma), c(5, 5))
  
  setWeights(model, list(NULL, NULL, matrix(1.0, 5, 5)))
  testthat::expect_equal(mean(model$model$env_weights[[1]]), 5)
  testthat::expect_equal(mean(model$model$spatial_weights[[1]]), 6)
  testthat::expect_equal(mean(model$model$get_sigma), 1)
  testthat::expect_equal(dim(model$model$env_weights[[1]]), c(5, 4))
  testthat::expect_equal(dim(model$model$spatial_weights[[1]]), c(5, 3))
  testthat::expect_equal(dim(model$model$get_sigma), c(5, 5))
  
})

