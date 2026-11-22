source("utils.R")

# Regressions from the 2026-10-15 adversarial review (task #10 in CLAUDE.md, HISTORY.md).

adv_data = function(n = 150L, J = 4L, seed = 3L) {
  set.seed(seed)
  X = data.frame(X1 = rnorm(n), X2 = rnorm(n))
  XY = data.frame(c1 = runif(n), c2 = runif(n))
  eta = 0.2 + as.matrix(X) %*% matrix(rnorm(2 * J, 0, 0.8), 2, J) +
    matrix(rnorm(n * 2), n, 2) %*% matrix(rnorm(2 * J, 0, 0.7), 2, J)
  list(X = X, XY = XY, Y = matrix(rbinom(n * J, 1, pnorm(eta)), n, J))
}

adv_fit = function(d, spatial = FALSE, Y = d$Y, ...) {
  sjSDM(Y = Y, env = linear(d$X, ~ X1 + X2),
        spatial = if (spatial) linear(d$XY, ~ 0 + c1:c2) else NULL,
        iter = 30L, sampling = 50L, device = "cpu", verbose = FALSE, seed = 1L, ...)
}

mcfadden = function(an) an$to_print$all$`R2 McFadden`

testthat::test_that("anova fractions add up to the full model", {
  skip_if_no_torch()
  testthat::skip_on_cran()
  d = adv_data()
  an = anova(adv_fit(d), samples = 300L, verbose = FALSE)
  r = mcfadden(an)
  testthat::expect_equal(sum(r[1:3]), r[4], tolerance = 1e-6)
  testthat::expect_equal(sum(an$species$Deviance$Full), an$results$Deviance[4], tolerance = 1e-6)
  testthat::expect_true(is.na(attr(anova(adv_fit(d), samples = 100L, verbose = FALSE,
                                         proposal = "prior")$ess, "newton_decrement")[1]))

  an = anova(adv_fit(d, spatial = TRUE), samples = 300L, verbose = FALSE)
  r = mcfadden(an)
  testthat::expect_equal(sum(r[1:7]), r[8], tolerance = 1e-6)
  eq = an$to_print$equal$`R2 McFadden`
  testthat::expect_equal(sum(eq[1:3]), eq[4], tolerance = 1e-6)
})

testthat::test_that("anova refuses a diagonal parent and survives NA in Y", {
  skip_if_no_torch()
  testthat::skip_on_cran()
  d = adv_data()
  testthat::expect_error(anova(adv_fit(d, biotic = bioticStruct(diag = TRUE)), verbose = FALSE),
                         "diag")
  Y = d$Y
  Y[c(3, 40, 77, 151, 400)] = NA
  m = adv_fit(d, Y = Y)
  testthat::expect_false(anyNA(anova(m, samples = 100L, verbose = FALSE)$to_print$all))
  testthat::expect_true(is.finite(Rsquared(m, verbose = FALSE)))
})

testthat::test_that("the null and the no-association sub-models carry no latent term", {
  skip_if_no_torch()
  testthat::skip_on_cran()
  set.seed(4)
  n = 200L
  X = data.frame(X1 = rnorm(n))
  Y = matrix(rpois(n * 3, exp(0.5 + 0.4 * X$X1)), n, 3)
  m = sjSDM(Y, env = linear(X, ~ X1), family = poisson("log"), iter = 100L, sampling = 50L,
            device = "cpu", verbose = FALSE, seed = 1L)
  glm_null = sum(apply(Y, 2, function(y) stats::logLik(stats::glm(y ~ 1, family = poisson()))))
  testthat::expect_lt(abs(sum(sjSDM:::get_null_ll(m, verbose = FALSE)) - glm_null), 1)
})

testthat::test_that("a fit leaves the caller's random stream alone", {
  skip_if_no_torch()
  d = adv_data(n = 40L)
  set.seed(123)
  ref = runif(2)
  set.seed(123)
  m = sjSDM(d$Y, env = linear(d$X, ~ X1), iter = 2L, verbose = FALSE, device = "cpu", seed = 1L)
  testthat::expect_identical(runif(2), ref)
})

testthat::test_that("rows of Y and the design have to match", {
  skip_if_no_torch()
  d = adv_data(n = 40L)
  X = d$X
  X$X1[1:5] = NA
  testthat::expect_error(sjSDM(d$Y, env = linear(X, ~ X1), iter = 2L, verbose = FALSE), "rows")
  testthat::expect_error(sjSDM(rbind(d$Y, d$Y[1:5, ]), env = linear(d$X, ~ X1), iter = 2L,
                               verbose = FALSE), "rows")
})

testthat::test_that("random blocks: NA newdata, nested numeric codes, poly() on the left", {
  skip_if_no_torch()
  set.seed(7)
  n = 120L
  X = data.frame(x = rnorm(n), z = rnorm(n), a = rep(1:4, each = 30), b = rep(1:3, 40))
  Y = matrix(rbinom(n * 3, 1, 0.4), n, 3)
  m = sjSDM(Y, env = linear(X, ~ x + (1 | a/b)), iter = 5L, verbose = FALSE, device = "cpu")
  testthat::expect_identical(vapply(m$settings$env$re, function(b) b$nlevels, 1L), c(12L, 4L))

  m = sjSDM(Y, env = linear(X, ~ x + (1 | a)), iter = 5L, verbose = FALSE, device = "cpu")
  nd = X
  nd$x[3] = NA
  p0 = predict(m, newdata = X, type = "raw")
  p = predict(m, newdata = nd, type = "raw")
  testthat::expect_identical(nrow(p), n)
  testthat::expect_true(all(is.na(p[3, ])))
  testthat::expect_equal(p[-3, ], p0[-3, ], tolerance = 1e-6)

  m = sjSDM(Y, env = linear(X, ~ x + (0 + poly(z, 2) | a)), iter = 5L, verbose = FALSE,
            device = "cpu")
  k = 1:40
  testthat::expect_equal(predict(m, newdata = X[k, ], type = "raw"),
                         predict(m, newdata = X, type = "raw")[k, ], tolerance = 1e-6)
})

testthat::test_that("formulas keep their own environment", {
  skip_if_no_torch()
  X = data.frame(x = rnorm(30), z = rnorm(30))
  hidden = 7L
  alpha = 10
  testthat::expect_identical(linear(X, ~ x + NN(z, hidden = hidden))$nn[[1]]$hidden, 7L)
  testthat::expect_equal(unname(linear(X, ~ 0 + I(x * alpha))$X[, 1]), 10 * X$x)
  f = function(h) linear(X, ~ x + NN(z, hidden = h))
  testthat::expect_identical(f(4L)$nn[[1]]$hidden, 4L)
})

testthat::test_that("getWeights/setWeights round trip carries random blocks and theta", {
  skip_if_no_torch()
  set.seed(8)
  X = data.frame(x = rnorm(80), g = factor(sample(letters[1:5], 80, TRUE)))
  Y = matrix(rbinom(80 * 3, 1, 0.5), 80, 3)
  fit = function(seed) sjSDM(Y, env = linear(X, ~ x + (1 | g)), iter = 5L, verbose = FALSE,
                             device = "cpu", seed = seed)
  m1 = fit(1L)
  m2 = fit(2L)
  setWeights(m2, getWeights(m1))
  testthat::expect_equal(predict(m2, type = "raw"), predict(m1, type = "raw"), tolerance = 1e-6)
})

# Regressions from the 2026-11-12 pre-release review.

testthat::test_that("simulate() carries the associations and covers all families", {
  skip_if_no_torch()
  set.seed(9)
  n = 300L
  z = rnorm(n)
  Y = 1 * (outer(z, rep(1.5, 4)) + matrix(rnorm(n * 4), n, 4) > 0)
  m = sjSDM(Y, env = linear(data.frame(x = rnorm(n)), ~ x), iter = 60L, verbose = FALSE,
            device = "cpu", seed = 1L)
  sim = simulate(m, nsim = 2L, seed = 1L)
  testthat::expect_identical(dim(sim), c(2L, n, 4L))
  R = stats::cor(sim[1, , ])
  testthat::expect_gt(mean(R[upper.tri(R)]), 0.2)
  for (fam in list(gaussian(), poisson(), "nbinom")) {
    Yc = if (identical(fam, "nbinom") || inherits(fam, "family") && fam$family == "poisson")
      matrix(rpois(n * 3, 2), n, 3) else matrix(rnorm(n * 3), n, 3)
    mf = sjSDM(Yc, env = linear(data.frame(x = rnorm(n)), ~ x), family = fam, iter = 5L,
               verbose = FALSE, device = "cpu")
    testthat::expect_identical(dim(simulate(mf)), c(1L, n, 3L))
  }
})

testthat::test_that("residuals are on the response scale", {
  skip_if_no_torch()
  d = adv_data(n = 60L)
  m = sjSDM(d$Y, env = linear(d$X, ~ X1), iter = 5L, verbose = FALSE, device = "cpu")
  r = residuals(m)
  testthat::expect_true(all(abs(r) <= 1))
  testthat::expect_equal(r, d$Y - predict(m))
})

testthat::test_that("a reloaded model is evaluated without dropout", {
  skip_if_no_torch()
  d = adv_data(n = 80L)
  m = sjSDM(d$Y, env = DNN(d$X, ~ X1 + X2, hidden = c(10L, 10L), dropout = 0.3), iter = 5L,
            verbose = FALSE, device = "cpu")
  f = tempfile(fileext = ".RDS")
  saveRDS(m, f)
  m2 = sjSDM:::checkModel(readRDS(f))
  testthat::expect_true(m2$model$net$training)
  m2$model$logLik(sjSDM:::config_X(m2$settings$env), m2$data$Y)
  testthat::expect_false(m2$model$net$training)
})

testthat::test_that("cross-validation AUC leaves one-class species out", {
  Y = cbind(c(0, 1, 0, 1), c(0, 0, 0, 0), c(1, NA, 0, 1))
  pred = cbind(c(0.1, 0.9, 0.2, 0.8), c(0.5, 0.5, 0.5, 0.5), c(0.9, 0.5, 0.1, 0.8))
  a = sjSDM:::species_auc(Y, pred)
  testthat::expect_equal(a, c(1, NA, 1))
  testthat::expect_equal(sjSDM:::macro_auc(a, Y), 1)
})

testthat::test_that("simulate_SDM follows the response and the link", {
  p = simulate_SDM(sites = 200L, species = 3L, link = "probit", seed = 3L)$response
  l = simulate_SDM(sites = 200L, species = 3L, link = "logit", seed = 3L)$response
  testthat::expect_false(identical(p, l))
  testthat::expect_true(all(l %in% c(0, 1)))
  cnt = simulate_SDM(sites = 50L, species = 3L, link = "probit", response = "count", seed = 3L)$response
  testthat::expect_true(any(cnt > 1))
  testthat::expect_error(simulate_SDM(link = "identical", response = "pa"), "presence-absence")
})

# Regressions from the 2026-11-18 NEWS-completeness review.

testthat::test_that("predict(Y = ) conditions without newdata, update() takes a seed", {
  skip_if_no_torch()
  set.seed(11)
  n = 150L
  z = rnorm(n)
  Y = 1 * (outer(z, rep(1.5, 4)) + matrix(rnorm(n * 4), n, 4) > 0)
  X = data.frame(x = rnorm(n))
  m = sjSDM(Y, env = linear(X, ~ x), iter = 40L, verbose = FALSE, device = "cpu", seed = 1L)
  Yc = Y
  Yc[, 1] = NA
  cond = predict(m, Y = Yc, sampling = 2000L)
  uncond = predict(m)
  testthat::expect_gt(max(abs(cond[, 1] - uncond[, 1])), 0.1)
  testthat::expect_gt(stats::cor(cond[, 1], Y[, 2]), stats::cor(uncond[, 1], Y[, 2]))
  testthat::expect_s3_class(update(m, seed = 3L), "sjSDM")
})

testthat::test_that("conditional expected abundance matches a brute-force reference", {
  skip_if_no_torch()
  set.seed(12)
  n = 200L
  z = rnorm(n)
  lam = exp(outer(rep(0.5, n), rep(1, 3)) + outer(z, c(0.8, 0.7, -0.6)))
  for (fam in list(poisson(), "nbinom")) {
    Y = matrix(rpois(length(lam), lam), n, 3)
    m = sjSDM(Y, env = linear(data.frame(x = rnorm(n)), ~ x), family = fam, iter = 60L,
              verbose = FALSE, device = "cpu", seed = 1L)
    Yc = Y
    Yc[, 1] = NA
    cond = predict(m, Y = Yc, sampling = 5000L)
    mu = predict(m, type = "raw")
    sig = m$model$get_sigma
    th = m$model$get_theta
    lik = if (identical(fam, "nbinom")) {
      r = 1 / (log1p(exp(th)) + 1e-4)
      function(y, E, j) stats::dnbinom(y, size = r[j], mu = E, log = TRUE)
    } else function(y, E, j) stats::dpois(y, E, log = TRUE)
    Z = matrix(rnorm(2e5 * ncol(sig)), ncol = ncol(sig))
    for (i in c(1L, 7L, 50L)) {
      lw = lik(Y[i, 2], exp(mu[i, 2] + Z %*% sig[2, ]), 2) + lik(Y[i, 3], exp(mu[i, 3] + Z %*% sig[3, ]), 3)
      w = exp(lw - max(lw))
      ref = sum(w * exp(mu[i, 1] + Z %*% sig[1, ])) / sum(w)
      testthat::expect_equal(cond[i, 1], ref, tolerance = 0.05)
    }
  }
})

testthat::test_that("conditional predictions: exact gaussian, any NA pattern, loo in one pass", {
  skip_if_no_torch()
  set.seed(3)
  n = 200L
  J = 4L
  X = data.frame(x = rnorm(n))
  L = matrix(c(0.8, 0.7, -0.6, 0.5, 0.2, -0.3, 0.4, 0.6), J, 2)
  Y = outer(X$x, rnorm(J)) + matrix(rnorm(n * 2), n, 2) %*% t(L) + matrix(rnorm(n * J, 0, 0.5), n, J)
  m = sjSDM(Y, env = linear(X, ~ x), family = gaussian(), iter = 60L, verbose = FALSE,
            device = "cpu", seed = 1L)
  mu = predict(m, type = "raw")
  sig = m$model$get_sigma
  V = sig %*% t(sig) + diag(exp(m$model$get_theta)^2)
  Yc = Y
  Yc[1:50, 1] = NA
  Yc[26:75, c(2, 4)] = NA
  p = predict(m, Y = Yc, sampling = 3000L)
  testthat::expect_identical(dim(p), c(n, 3L))
  testthat::expect_true(all(is.na(p[76:n, ])))
  for (i in c(1L, 30L, 60L)) {
    t = which(is.na(Yc[i, ]))
    o = which(!is.na(Yc[i, ]))
    ex = mu[i, t] + V[t, o, drop = FALSE] %*% solve(V[o, o]) %*% (Y[i, o] - mu[i, o])
    testthat::expect_equal(unname(p[i, match(t, c(1L, 2L, 4L))]), as.numeric(ex), tolerance = 0.03)
  }
  lo = predict(m, Y = Y, loo = TRUE, sampling = 2000L)
  testthat::expect_identical(dim(lo), c(n, J))
  testthat::expect_identical(dim(attr(lo, "ess")), c(n, J))
  for (i in c(5L, 80L, 150L)) {
    ex = mu[i, 3] + V[3, -3, drop = FALSE] %*% solve(V[-3, -3]) %*% (Y[i, -3] - mu[i, -3])
    testthat::expect_equal(lo[i, 3], as.numeric(ex), tolerance = 0.05)
  }
  testthat::expect_error(predict(m, Y = Yc, loo = TRUE), "complete")
})

testthat::test_that("jit = TRUE trains the same model, NA cells stay masked after the trace", {
  skip_if_no_torch()
  set.seed(1)
  n = 103
  X = data.frame(a = rnorm(n))
  Y = 1 * (matrix(rnorm(n * 5), n) + X$a > 0)
  Y[60:103, 2] = NA
  fit = function(jit) sjSDM(Y, env = linear(X, ~.), iter = 5L, step_size = 20L, seed = 3L,
                            verbose = FALSE, se = FALSE, control = sjSDMControl(jit = jit))
  a = fit(FALSE)
  b = fit(TRUE)
  testthat::expect_equal(unlist(getWeights(b)), unlist(getWeights(a)), tolerance = 1e-5)
  testthat::expect_equal(b$history$train_l, a$history$train_l, tolerance = 1e-5)
})

testthat::test_that("default df is half the species, capped at a tenth of the sites", {
  skip_if_no_torch()
  set.seed(1)
  fit = function(n, J) {
    Y = matrix(rbinom(n * J, 1, 0.4), n, J)
    sjSDM(Y, env = linear(data.frame(a = rnorm(n)), ~.), iter = 1L, verbose = FALSE, se = FALSE)
  }
  testthat::expect_identical(fit(100L, 60L)$loss_properties$df, 10L)
  testthat::expect_identical(fit(400L, 30L)$loss_properties$df, 15L)
  testthat::expect_identical(fit(30L, 6L)$loss_properties$df, 5L)
})

testthat::test_that("conditional predictions keep newdata rows with a missing predictor as NA", {
  skip_if_no_torch()
  set.seed(1)
  X = data.frame(x = rnorm(120))
  Y = matrix(rbinom(120 * 6, 1, 0.5), 120)
  m = sjSDM(Y, env = linear(X, ~.), iter = 5L, verbose = FALSE, se = FALSE)
  nd = X[1:10, , drop = FALSE]
  nd$x[2] = NA
  Yq = Y[1:10, ]
  Yq[, 1] = NA
  p = predict(m, newdata = nd, Y = Yq)
  testthat::expect_true(is.na(p[2, 1]) && is.na(attr(p, "ess")[2]))
  testthat::expect_false(anyNA(p[-2, 1]))
})
