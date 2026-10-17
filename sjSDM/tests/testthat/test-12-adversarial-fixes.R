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
