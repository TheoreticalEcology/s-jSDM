source("utils.R")

# Regression gate for the backend rewrite. Every assertion here passes on the pre-rewrite
# backend; each one covers a behaviour change that produces no error and no other failing
# test. See BACKEND_rewrite_plan.md section 8.1.

fit_small = function() {
  set.seed(3)
  com = simulate_SDM(env = 3L, species = 5L, sites = 80L)
  sjSDM(Y = com$response, env = com$env_weights, iter = 3L, device = "cpu", verbose = FALSE)
}

# checkModel() decides whether to rebuild from is.matrix(m$get_sigma). R6 returns NULL for a
# missing member instead of erroring, so dropping the binding makes a live model look dead and
# every predict() rebuilds -- and the rebuild calls sjsdm_set_seed(), resetting the global RNG.
testthat::test_that("a freshly fitted model is alive and predict() leaves the RNG alone", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  m = fit_small()

  testthat::expect_true(sjSDM:::model_is_alive(m$model))
  testthat::expect_true(is.matrix(m$model$get_sigma))
  testthat::expect_equal(nrow(m$model$get_sigma), ncol(m$data$Y))

  # the comparison must be before vs after. Two post-predict states agree even on a dead
  # model, because every rebuild resets to the same seed.
  set.seed(1)
  runif(1)
  before = .Random.seed
  before_torch = as.numeric(torch::torch_get_rng_state()$cpu())
  p1 = predict(m)
  testthat::expect_identical(.Random.seed, before)
  p2 = predict(m)
  testthat::expect_identical(.Random.seed, before)
  testthat::expect_identical(as.numeric(torch::torch_get_rng_state()$cpu()), before_torch)
  testthat::expect_equal(p1, p2)
})

# anova.R:333-337 takes a raw slice of the sigma tensor and reads its dtype/device, so sigma
# has to stay a leaf tensor reachable from outside the backend object.
testthat::test_that("sigma stays a leaf tensor that can be sliced from outside", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  m = fit_small()
  s = m$model$sigma

  testthat::expect_true(inherits(s, "torch_tensor"))
  testthat::expect_true(s$requires_grad)
  testthat::expect_true(s$is_leaf)
  testthat::expect_equal(s$shape, c(ncol(m$data$Y), ncol(m$model$get_sigma)))

  sub = s$detach()[1, , drop = FALSE]
  testthat::expect_equal(sub$shape, c(1L, s$shape[2]))
  testthat::expect_true(inherits(m$model$dtype, "torch_dtype"))
  testthat::expect_true(inherits(m$model$device, "torch_device"))
  testthat::expect_true(inherits(sub$dtype, "torch_dtype"))
  testthat::expect_true(inherits(sub$device, "torch_device"))
  testthat::expect_equal(m$model$alpha, 1.70169)
  testthat::expect_identical(m$model$link, "probit")
})

# The latent factor is integrated by an equal-weight average over `noise`, so feeding the
# stratified inverse-CDF grid turns the Monte Carlo into a quadrature rule and the comparison
# against integrate() becomes exact rather than noisy. Measured: this grid reproduces
# integrate() to 1e-6, while 20000 random draws have a per-site sd of 0.007. The case uses a
# non-identity alpha and a non-zero sigma, so it pins mu + z %*% t(sigma), the alpha scaling,
# the probability guard and the logsumexp reduction together.
testthat::test_that("the latent-factor integration matches exact quadrature", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  tt = function(x) sjSDM:::sjsdm_tensor(x)
  n = 6L
  sp = 2L
  alpha = 1.3
  mu = matrix(c(-1.2, 0.4, 0.9, -0.3, 0.1, 1.5,
                 0.7, -0.8, 0.2, 1.1, -1.4, 0.6), n, sp)
  sigma = matrix(c(0.8, -1.3), sp, 1L)
  Y = matrix(c(1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0), n, sp)

  want = sapply(1:n, function(i) {
    f = function(z) sapply(z, function(zz) {
      p = stats::plogis(alpha * (mu[i, ] + zz * sigma[, 1])) * 0.999999 + 0.0000005
      exp(sum(Y[i, ] * log(p) + (1 - Y[i, ]) * log(1 - p))) * stats::dnorm(zz)
    })
    -log(stats::integrate(f, -Inf, Inf, rel.tol = 1e-12, subdivisions = 2000L)$value)
  })

  S = 20000L
  noise = array(0, dim = c(S, n, 1L))
  for (i in 1:n) noise[, i, 1] = stats::qnorm((seq_len(S) - 0.5) / S)
  got = as.numeric(sjSDM:::mvp_logLik(tt(mu), tt(Y), tt(sigma), link = "logit", alpha = alpha,
                                      noise = torch::torch_tensor(noise, dtype = torch::torch_float32()))$cpu())
  testthat::expect_lt(max(abs(got - want)), 1e-5)

  # the default path draws its own noise, so it also has to hit the same value, now within
  # Monte-Carlo error (per-site sd 0.007 at this sampling)
  mc = as.numeric(sjSDM:::mvp_logLik(tt(mu), tt(Y), tt(sigma), link = "logit", alpha = alpha,
                                     sampling = S)$cpu())
  testthat::expect_lt(max(abs(mc - want)), 0.05)
})

# Two species with a rank-2 sigma: the joint probability is an exact bivariate orthant
# probability of N(mu, sigma sigma' + I), which mvtnorm::pmvnorm gives to 1e-9. Measured on
# this case: the 300 x 300 grid reproduces pmvnorm to 3.7e-4 when the exact normal CDF is
# used, and the shipped sigmoid(1.70169 x) approximation adds 0.0205 nats on top. The
# tolerance below is therefore an integration gate, not an accuracy claim.
testthat::test_that("the two-species probit likelihood matches mvtnorm::pmvnorm", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  tt = function(x) sjSDM:::sjsdm_tensor(x)
  n = 6L
  sp = 2L
  mu = matrix(c(-1.2, 0.4, 0.9, -0.3, 0.1, 1.5,
                 0.7, -0.8, 0.2, 1.1, -1.4, 0.6), n, sp)
  sigma = matrix(c(0.9, 0.3, -0.6, 0.7), sp, 2L)
  Y = matrix(c(1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0), n, sp)
  S2 = sigma %*% t(sigma) + diag(sp)

  want = sapply(1:n, function(i) {
    -log(mvtnorm::pmvnorm(lower = ifelse(Y[i, ] == 1, 0, -Inf),
                          upper = ifelse(Y[i, ] == 1, Inf, 0),
                          mean = mu[i, ], sigma = S2,
                          algorithm = mvtnorm::GenzBretz(abseps = 1e-9, maxpts = 1e6))[1])
  })

  k = 300L
  g = as.matrix(expand.grid(stats::qnorm((seq_len(k) - 0.5) / k),
                            stats::qnorm((seq_len(k) - 0.5) / k)))
  noise = array(0, dim = c(nrow(g), n, 2L))
  for (i in 1:n) noise[, i, ] = g
  got = as.numeric(sjSDM:::mvp_logLik(tt(mu), tt(Y), tt(sigma), link = "probit", alpha = 1.70169,
                                      noise = torch::torch_tensor(noise, dtype = torch::torch_float32()))$cpu())
  testthat::expect_lt(max(abs(got - want)), 0.03)

  # the same quadrature with the exact normal CDF, which isolates the approximation from the
  # integration. Its size is a deliberate property of the backend (plan section 6.4), so the
  # band fails both ways: a broken integration widens it, replacing sigmoid by pnorm closes it.
  exact = sapply(1:n, function(i) {
    p = stats::pnorm(sweep(g %*% t(sigma), 2, mu[i, ], "+")) * 0.999999 + 0.0000005
    -log(mean(exp(colSums(Y[i, ] * t(log(p)) + (1 - Y[i, ]) * t(log(1 - p))))))
  })
  testthat::expect_lt(max(abs(exact - want)), 1e-3)
  bias = max(abs(got - exact))
  testthat::expect_gt(bias, 0.01)
  testthat::expect_lt(bias, 0.03)
})

# test-4's reload test compares dim() only, so a rebuild that restores the wrong weights
# passes it. The default probit marginal prediction is closed-form, hence bit-deterministic.
testthat::test_that("a saved and reloaded model reproduces its values, not just its shapes", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  m = fit_small()
  p = predict(m)
  cv = getCov(m)
  cf = coef(m)
  sg = m$model$get_sigma

  f = tempfile(fileext = ".RDS")
  saveRDS(m, f)
  m2 = readRDS(f)
  unlink(f)

  testthat::expect_false(sjSDM:::model_is_alive(m2$model))
  m2 = sjSDM:::checkModel(m2)
  testthat::expect_true(sjSDM:::model_is_alive(m2$model))

  testthat::expect_equal(predict(m2), p, tolerance = 1e-6)
  testthat::expect_equal(getCov(m2), cv, tolerance = 1e-6)
  testthat::expect_equal(coef(m2), cf, tolerance = 1e-6)
  testthat::expect_equal(m2$model$get_sigma, sg, tolerance = 1e-6)
})
