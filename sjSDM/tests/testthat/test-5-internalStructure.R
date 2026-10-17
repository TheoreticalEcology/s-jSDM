source("utils.R")

# internalStructure() replaced the deprecated importance(). It is exported, it is what the
# metacommunity-structure figures are built from, and it had no tests at all.

cache = new.env()

get_anova = function() {
  if (is.null(cache$a)) {
    set.seed(42)
    n = 60L; sp = 5L
    X = matrix(rnorm(n * 2), n, 2)
    SP = matrix(rnorm(n * 2), n, 2)
    Y = matrix(rbinom(n * sp, 1, 0.4), n, sp)
    m = sjSDM(Y = Y, env = linear(X, ~.), spatial = linear(SP, ~0 + .), iter = 10L,
              sampling = 50L, verbose = FALSE, seed = 3L, device = is_gpu_available())
    cache$a = anova(m, samples = 100L, verbose = FALSE)
    cache$n = n
    cache$sp = sp
  }
  cache$a
}

testthat::test_that("internalStructure returns one row per site and per species", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  s = internalStructure(get_anova())
  testthat::expect_s3_class(s, "sjSDMinternalStructure")
  testthat::expect_named(s, c("raws", "internals", "Rsquared", "fractions", "anova"))
  for (part in c("raws", "internals")) {
    testthat::expect_equal(nrow(s[[part]]$Sites), cache$n)
    testthat::expect_equal(nrow(s[[part]]$Species), cache$sp)
    testthat::expect_named(s[[part]]$Sites, c("env", "spa", "codist", "r2"))
    testthat::expect_named(s[[part]]$Species, c("env", "spa", "codist", "r2"))
  }
  testthat::expect_true(all(is.finite(as.matrix(s$internals$Sites))))
  testthat::expect_true(all(is.finite(as.matrix(s$internals$Species))))
})

testthat::test_that("every fractions and Rsquared option runs", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  a = get_anova()
  for (fr in c("mvp_proportional", "mvp", "discard", "proportional", "equal")) {
    for (r2 in c("McFadden", "Nagelkerke")) {
      s = internalStructure(a, Rsquared = r2, fractions = fr)
      testthat::expect_equal(s$fractions, fr)
      testthat::expect_equal(s$Rsquared, r2)
      testthat::expect_equal(dim(s$internals$Sites), c(cache$n, 4L))
    }
  }
  testthat::expect_error(internalStructure(a, fractions = "garbage"))
})

testthat::test_that("the proportional fractions add up to the model R squared", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  # only the two proportional schemes redistribute the whole shared fraction, so only they
  # have to reconstitute R2 exactly. mvp, equal and discard deliberately do not.
  a = get_anova()
  for (fr in c("mvp_proportional", "proportional")) {
    s = internalStructure(a, fractions = fr, negatives = "raw")
    for (part in c("Sites", "Species")) {
      d = s$raws[[part]]
      testthat::expect_equal(d$env + d$spa + d$codist, d$r2, tolerance = 1e-10)
    }
  }
})

testthat::test_that("negatives are floored, rescaled or left alone", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  a = get_anova()
  raw = internalStructure(a, negatives = "raw")
  floored = internalStructure(a, negatives = "floor")
  scaled = internalStructure(a, negatives = "scale")
  for (part in c("Sites", "Species")) {
    r = as.matrix(raw$internals[[part]][, 1:3])
    testthat::expect_equal(as.matrix(floored$internals[[part]][, 1:3]),
                           pmin(pmax(r, 0), Inf))
    testthat::expect_true(all(as.matrix(scaled$internals[[part]][, 1:3]) >= 0))
    testthat::expect_true(all(as.matrix(scaled$internals[[part]][, 1:3]) <= 1))
    # the raw fractions are kept untouched whatever the standardisation
    testthat::expect_equal(raw$raws[[part]], floored$raws[[part]])
  }
  testthat::expect_equal(raw$internals$Sites$r2, floored$internals$Sites$r2)
})

testthat::test_that("internalStructure refuses a non-spatial model", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  set.seed(7)
  X = matrix(rnorm(50 * 2), 50, 2)
  Y = matrix(rbinom(50 * 4, 1, 0.4), 50, 4)
  m = sjSDM(Y = Y, env = linear(X, ~.), iter = 5L, sampling = 50L, verbose = FALSE, seed = 2L)
  a = anova(m, samples = 50L, verbose = FALSE)
  testthat::expect_error(internalStructure(a), "only supported for spatial models")
})

testthat::test_that("print and plot methods work", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  s = internalStructure(get_anova())
  testthat::expect_equal(print(s), s$internals)
  testthat::expect_error({ .k = testthat::capture_output(plot(s)) }, NA)
  testthat::expect_error({ .k = testthat::capture_output(plot(s, negatives = "scale")) }, NA)
})

testthat::test_that("plotAssemblyEffects needs predictors for the species response", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  s = internalStructure(get_anova())
  testthat::expect_error(plotAssemblyEffects(s, response = "species"),
                         "Species response requires predictors")
  testthat::expect_error({ .k = testthat::capture_output(plotAssemblyEffects(s)) }, NA)
  testthat::expect_error({
    .k = testthat::capture_output(plotAssemblyEffects(s, pred = rnorm(cache$n)))
  }, NA)
})
