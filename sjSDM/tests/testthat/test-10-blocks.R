source("utils.R")

# NN() blocks inside the environmental formula. PREDICTOR_plan.md section 10.1.

block_data = function(n = 60L) {
  set.seed(11)
  data.frame(temp = rnorm(n), soil = rnorm(n), ndvi = rnorm(n),
             g = factor(sample(letters[1:3], n, TRUE)))
}

block_Y = function(n = 60L, sp = 4L) {
  set.seed(12)
  matrix(rbinom(n * sp, 1, 0.4), n, sp)
}

# linear() reads its formula out of match.call(), so a formula held in a variable has to go in
# through do.call() -- see the wart list in CLAUDE.md
env_block = function(f, X = block_data()) do.call(linear, list(data = X, formula = f))

fit_blocks = function(f, X = block_data(), Y = block_Y(), ...) {
  sjSDM(Y = Y, env = env_block(f, X), iter = 3L, sampling = 20L, verbose = FALSE,
        seed = 5L, device = "cpu", ...)
}

# one tensor per block, evaluated outside the batching predict() uses
block_out = function(m, k) {
  cfg = m$settings$env
  X = sjSDM:::config_X(cfg)[[k]]
  as.matrix(m$model$net$blocks[[k]](sjSDM:::sjsdm_tensor(X, m$model$dtype,
                                                        m$model$device))$cpu())
}

testthat::test_that("a formula without NN() gives one block and the old design", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  X = block_data()
  e = env_block(~ temp + soil, X)
  old = sjSDM:::design(~ temp + soil, X)

  testthat::expect_length(e$nn, 0L)
  testthat::expect_identical(e$X, old$X)
  testthat::expect_identical(e$intercept, old$intercept)

  m = fit_blocks(~ temp + soil)
  testthat::expect_length(m$model$net$blocks, 1L)
  testthat::expect_identical(m$names, colnames(old$X))
})

testthat::test_that("the parse keeps the parametric design on the config", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  e = env_block(~ temp + NN(soil + ndvi, hidden = c(5L, 5L)))

  testthat::expect_identical(colnames(e$X), c("(Intercept)", "temp"))
  testthat::expect_length(e$nn, 1L)
  testthat::expect_identical(colnames(e$nn[[1]]$X), c("soil", "ndvi"))
  testthat::expect_true(e$intercept)
  testthat::expect_false(e$nn[[1]]$intercept)
  testthat::expect_identical(e$nn[[1]]$hidden, c(5L, 5L))
})

testthat::test_that("every form of section 3 parses to the blocks it names", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  X = block_data()
  cases = list(list(~ temp,                               2L, 0L, NULL),
               list(~ temp + NN(soil),                    2L, 1L, "soil"),
               list(~ NN(soil + ndvi),                    1L, 1L, c("soil", "ndvi")),
               list(~ 0 + NN(.),                          0L, 1L, NULL),
               list(~ NN(temp) + NN(soil),                1L, 2L, "temp"),
               list(~ poly(temp, 2) + NN(soil),           3L, 1L, "soil"),
               list(~ temp + NN(hidden = 4L, soil),       2L, 1L, "soil"),
               list(~ temp:soil + NN(ndvi),               2L, 1L, "ndvi"),
               list(~ temp + NN(g),                       2L, 1L, c("ga", "gb", "gc")))
  for (k in cases) {
    e = env_block(k[[1]], X)
    testthat::expect_identical(ncol(e$X), k[[2]], info = deparse(k[[1]]))
    testthat::expect_length(e$nn, k[[3]])
    if (!is.null(k[[4]]))
      testthat::expect_identical(colnames(e$nn[[1]]$X), k[[4]], info = deparse(k[[1]]))
  }
  # NN(hidden = 4L, soil) is match.call()-canonicalised, not read positionally
  testthat::expect_identical(env_block(~ temp + NN(hidden = 4L, soil), X)$nn[[1]]$hidden, 4L)
})

testthat::test_that("the rejected forms of section 3.4 give an informative error", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  X = block_data()
  testthat::expect_error(env_block(~ temp + NN(temp + soil), X), "cannot be in both")
  testthat::expect_error(env_block(~ temp + NN(.), X), "cannot be in both")
  testthat::expect_error(env_block(~ poly(temp, 2) + NN(temp), X), "cannot be in both")
  testthat::expect_error(env_block(~ NN(soil) + NN(soil + ndvi), X), "two NN\\(\\) blocks")
  testthat::expect_error(env_block(~ temp:NN(soil), X), "inside an interaction")
  testthat::expect_error(do.call(DNN, list(data = X, formula = ~ temp + NN(soil))),
                         "DNN\\(\\) is already a network")
  testthat::expect_error(NN(), "needs the covariates")
  testthat::expect_error(
    sjSDM(Y = block_Y(), env = env_block(~ temp, X), spatial = env_block(~ 0 + NN(soil), X),
          iter = 2L, verbose = FALSE),
    "not supported in the spatial formula")
})

testthat::test_that("the predictor is the sum of its blocks", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  m = fit_blocks(~ temp + NN(soil + ndvi, hidden = c(5L, 5L)))

  testthat::expect_length(m$model$net$blocks, 2L)
  raw = predict(m, type = "raw", batch_size = nrow(m$data$Y))
  testthat::expect_equal(raw, block_out(m, 1L) + block_out(m, 2L), tolerance = 1e-6)

  # section 4.2: the output layer of a block in a sum carries no bias
  nn = m$model$net$blocks[[2]]
  testthat::expect_null(nn[[length(nn$children)]]$bias)
  # ... and its first hidden layer does, there being no intercept column to supply one
  testthat::expect_false(is.null(nn[[1]]$bias))
})

testthat::test_that("a removed block contributes exactly zero", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  m = fit_blocks(~ temp + NN(soil + ndvi, hidden = c(5L, 5L)))

  no_nn = update(m, env_blocks = 1L, verbose = FALSE)
  testthat::expect_length(no_nn$settings$env$nn, 0L)
  testthat::expect_length(no_nn$model$net$blocks, 1L)
  testthat::expect_identical(predict(no_nn, type = "raw", batch_size = 60L), block_out(no_nn, 1L))

  no_par = update(m, env_blocks = 2L, verbose = FALSE)
  testthat::expect_identical(ncol(no_par$data$X), 0L)
  testthat::expect_true(all(block_out(no_par, 1L) == 0))
  testthat::expect_identical(predict(no_par, type = "raw", batch_size = 60L),
                             block_out(no_par, 2L))

  # the parametric case, which is the regression test for the DNN leak of section 8.3
  off = update(m, env_formula = ~ 0, verbose = FALSE)
  testthat::expect_true(all(predict(off, type = "raw") == 0))
})

testthat::test_that("predict(newdata) rebuilds every block from its stored terms", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  X = block_data()
  m = fit_blocks(~ poly(temp, 2) + NN(soil + ndvi, hidden = c(4L)), X = X)
  raw = predict(m, type = "raw", batch_size = 60L)

  # poly() re-derived from five rows gives a different basis, so this is the test that the
  # terms object and not the formula drives the newdata design
  testthat::expect_equal(predict(m, newdata = X[1:5, ], type = "raw", batch_size = 5L),
                         raw[1:5, ], tolerance = 1e-6)
  testthat::expect_length(sjSDM:::sjsdm_newdata(m$settings$env, X[1:5, ]), 2L)

  # a factor level absent from newdata must still produce the fitted block width
  nd = X[X$g == levels(X$g)[1], ][1:3, ]
  testthat::expect_identical(ncol(predict(m, newdata = nd, type = "raw", batch_size = 3L)),
                             ncol(m$data$Y))
})

testthat::test_that("a block model survives saveRDS with its values", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  m = fit_blocks(~ temp + NN(soil + ndvi, hidden = c(5L, 5L)))
  before = predict(m, type = "raw", batch_size = 60L)

  f = tempfile(fileext = ".RDS")
  on.exit(unlink(f))
  saveRDS(m, f)
  back = sjSDM:::checkModel(readRDS(f))

  testthat::expect_identical(predict(back, type = "raw", batch_size = 60L), before)
  testthat::expect_identical(coef(back), coef(m))
  testthat::expect_identical(back$model$get_sigma, m$model$get_sigma)
})

testthat::test_that("the block penalties are per block", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  free = fit_blocks(~ temp + NN(soil + ndvi, hidden = c(5L, 5L)))
  pen = fit_blocks(~ temp + NN(soil + ndvi, hidden = c(5L, 5L), lambda = 1.0))

  testthat::expect_identical(free$logLik[[2]], 0)
  testthat::expect_gt(pen$logLik[[2]], 0)
})

testthat::test_that("anova and summary keep working beside an NN block", {
  testthat::skip_on_cran()
  skip_if_no_torch()
  m = fit_blocks(~ temp + NN(soil + ndvi, hidden = c(5L, 5L)))

  s = summary(m)
  testthat::expect_identical(rownames(s$coefs), c("(Intercept)", "temp"))
  testthat::expect_true(grepl("NN block 1", m$env_architecture))

  a = anova(m, samples = 50L, verbose = FALSE)
  testthat::expect_true(all(c("F_A", "F_B", "Full", "Null") %in% a$results$models))
})
