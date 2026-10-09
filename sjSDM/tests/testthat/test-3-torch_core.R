source("utils.R")

# With sigma = 0 the multivariate probit collapses to independent species, so the
# Monte-Carlo likelihood must reproduce the univariate one exactly (every MC draw
# gives the same value). That is an analytic reference, no second backend needed.

testthat::test_that("MC likelihood collapses to the independent case", {
  skip_if_no_torch()
  set.seed(42)
  n = 30; sp = 4; df = 3
  mu = matrix(rnorm(n * sp), n, sp)
  sigma = matrix(0, sp, df)
  tt = function(x) sjSDM:::sjsdm_tensor(x)

  Y = matrix(rbinom(n * sp, 1, 0.5), n, sp)
  got = as.numeric(sjSDM:::mvp_logLik(tt(mu), tt(Y), tt(sigma), link = "logit",
                                      alpha = 1.0, sampling = 5L)$cpu())
  p = 1 / (1 + exp(-mu))
  p = p * 0.999999 + 0.0000005
  want = -rowSums(Y * log(p) + (1 - Y) * log(1 - p))
  testthat::expect_equal(got, want, tolerance = 1e-4)

  Yc = matrix(rpois(n * sp, 2), n, sp)
  got = as.numeric(sjSDM:::mvp_logLik(tt(mu), tt(Yc), tt(sigma), link = "count",
                                      sampling = 5L)$cpu())
  want = -rowSums(stats::dpois(Yc, exp(mu), log = TRUE))
  testthat::expect_equal(got, want, tolerance = 1e-4)

  Yg = matrix(rnorm(n * sp), n, sp)
  theta = torch::torch_zeros(sp)
  got = as.numeric(sjSDM:::mvp_logLik(tt(mu), tt(Yg), tt(sigma), link = "normal",
                                      sampling = 5L, theta = theta)$cpu())
  want = -rowSums(stats::dnorm(Yg, mu, 1, log = TRUE))
  testthat::expect_equal(got, want, tolerance = 1e-4)
})

testthat::test_that("the nbinom link matches dnbinom's mu parameterisation", {
  skip_if_no_torch()
  set.seed(11)
  n = 40; sp = 5; df = 3
  tt = function(x) sjSDM:::sjsdm_tensor(x)
  mu = matrix(rnorm(n * sp, sd = 0.5), n, sp)
  sigma = matrix(0, sp, df)
  Y = matrix(rnbinom(n * sp, mu = 2, size = 2), n, sp)
  theta = torch::torch_zeros(sp)

  got = as.numeric(sjSDM:::mvp_logLik(tt(mu), tt(Y), tt(sigma), link = "nbinom",
                                      sampling = 5L, theta = theta)$cpu())
  # the model's r is 1 / (softplus(theta) + eps), and dnbinom's size is that r
  r = 1 / (log(2) + 1e-4)
  want = -rowSums(stats::dnbinom(Y, size = r, mu = exp(mu), log = TRUE))
  testthat::expect_equal(got, want, tolerance = 1e-3)
})

testthat::test_that("the linear link is a guarded clamp", {
  skip_if_no_torch()
  set.seed(13)
  n = 30; sp = 4; df = 2
  tt = function(x) sjSDM:::sjsdm_tensor(x)
  mu = matrix(runif(n * sp, 0.05, 0.95), n, sp)
  sigma = matrix(0, sp, df)
  Y = matrix(rbinom(n * sp, 1, 0.5), n, sp)

  got = as.numeric(sjSDM:::mvp_logLik(tt(mu), tt(Y), tt(sigma), link = "linear",
                                      sampling = 5L)$cpu())
  p = mu * 0.999999 + 0.0000005
  want = -rowSums(Y * log(p) + (1 - Y) * log(1 - p))
  testthat::expect_equal(got, want, tolerance = 1e-4)

  # outside [0, 1] the response saturates, and there the guarded probability is only
  # float32-accurate: 0.999999 + 5e-7 rounds to 0.9999994636, so log(1 - E) is off by
  # 0.07 nats against the double-precision value. Compare against what float32 can hold.
  sat = as.numeric(sjSDM:::mvp_logLik(tt(matrix(c(-0.4, 1.4), 1, 2)), tt(matrix(c(1, 0), 1, 2)),
                                      tt(matrix(0, 2, df)), link = "linear", sampling = 5L)$cpu())
  E = torch::torch_tensor(c(0, 1), dtype = torch::torch_float32())$mul(0.999999)$add(0.0000005)
  testthat::expect_equal(sat, -sum(as.numeric(E$log()[1]), as.numeric((1 - E)$log()[2])),
                         tolerance = 1e-4)
})

testthat::test_that("the probability guard is applied to bounded links only", {
  skip_if_no_torch()
  eta = sjSDM:::sjsdm_tensor(matrix(c(-2, 0, 2, 4), 2, 2))
  guarded = function(link) as.numeric(sjSDM:::mvp_response(eta, link, guard = TRUE)$cpu())
  raw = function(link) as.numeric(sjSDM:::mvp_response(eta, link, guard = FALSE)$cpu())
  for (link in c("probit", "logit", "linear"))
    testthat::expect_equal(guarded(link), raw(link) * 0.999999 + 0.0000005, tolerance = 1e-7)
  # a Poisson rate and a Gaussian mean are not probabilities, so the guard would only bias them
  for (link in c("count", "nbinom", "normal"))
    testthat::expect_equal(guarded(link), raw(link))
  testthat::expect_error(sjSDM:::mvp_response(eta, "garbage"), "unknown link")
})

testthat::test_that("legacy_guard reproduces the python guard on unbounded links", {
  skip_if_no_torch()
  set.seed(17)
  n = 20; sp = 3; df = 2
  tt = function(x) sjSDM:::sjsdm_tensor(x)
  mu = matrix(rnorm(n * sp, sd = 0.5), n, sp)
  sigma = matrix(0, sp, df)
  Y = matrix(rpois(n * sp, 2), n, sp)
  got = as.numeric(sjSDM:::mvp_logLik(tt(mu), tt(Y), tt(sigma), link = "count", sampling = 5L,
                                      legacy_guard = TRUE)$cpu())
  want = -rowSums(stats::dpois(Y, exp(mu) * 0.999999 + 0.0000005, log = TRUE))
  testthat::expect_equal(got, want, tolerance = 1e-4)
})

testthat::test_that("weight_decay is applied with the parameter value, not its index", {
  skip_if_no_torch()
  # sjSDM's RMSprop and SGD are torch's ignite optimizers. The *R* implementations
  # (torch::optim_rmsprop / optim_sgd, torch 0.17.0) pass the loop index to add(alpha = )
  # instead of the parameter, and the default control is RMSprop(weight_decay = 1e-4), so
  # picking the wrong one silently changes every default fit. This pins the correct update.
  p0 = c(2, -3); lr = 0.1; wd = 0.5; g = rep(1, 2)
  step1 = function(f) {
    p = torch::torch_tensor(p0, requires_grad = TRUE)
    o = f(list(p)); o$zero_grad(); p$sum()$backward(); o$step()
    as.numeric(p$detach())
  }

  testthat::expect_equal(step1(sjSDM:::optimizer_SGD(lr = lr, weight_decay = wd)),
                         p0 - lr * (g + wd * p0), tolerance = 1e-6)

  gd = g + wd * p0
  testthat::expect_equal(step1(sjSDM:::optimizer_RMSprop(lr = lr, alpha = 0.99, eps = 1e-8,
                                                         weight_decay = wd)),
                         p0 - lr * gd / (sqrt(0.01 * gd^2) + 1e-8), tolerance = 1e-6)

  # weight_decay = 0 must leave the update untouched
  testthat::expect_equal(step1(sjSDM:::optimizer_SGD(lr = lr)), p0 - lr * g, tolerance = 1e-6)

  # and the R implementations are still wrong, which is why the ignite ones are used
  testthat::expect_false(isTRUE(all.equal(
    step1(function(pp) torch::optim_sgd(pp, lr = lr, weight_decay = wd)),
    p0 - lr * (g + wd * p0), tolerance = 1e-6)))
})

testthat::test_that("fixed noise makes the likelihood deterministic", {
  skip_if_no_torch()
  set.seed(1)
  n = 20; sp = 4; df = 2; S = 7
  tt = function(x) sjSDM:::sjsdm_tensor(x)
  mu = tt(matrix(rnorm(n * sp), n, sp))
  sigma = tt(matrix(rnorm(sp * df, sd = 0.5), sp, df))
  Y = tt(matrix(rbinom(n * sp, 1, 0.4), n, sp))
  noise = torch::torch_tensor(array(rnorm(S * n * df), c(S, n, df)),
                              dtype = torch::torch_float32())
  a = sjSDM:::mvp_logLik(mu, Y, sigma, "probit", 1.70169, S, noise = noise)
  b = sjSDM:::mvp_logLik(mu, Y, sigma, "probit", 1.70169, S, noise = noise)
  testthat::expect_equal(as.numeric(a$cpu()), as.numeric(b$cpu()))
})

testthat::test_that("NaN responses are dropped from the likelihood", {
  skip_if_no_torch()
  set.seed(3)
  n = 25; sp = 4; df = 2; S = 9
  tt = function(x) sjSDM:::sjsdm_tensor(x)
  mu = matrix(rnorm(n * sp), n, sp)
  sigma = matrix(0, sp, df)
  Y = matrix(rbinom(n * sp, 1, 0.5), n, sp)
  Yna = Y; Yna[, 1] = NA
  with_na = as.numeric(sjSDM:::mvp_logLik(tt(mu), tt(Yna), tt(sigma), "logit", 1.0, S)$cpu())
  dropped = as.numeric(sjSDM:::mvp_logLik(tt(mu[, -1]), tt(Y[, -1]), tt(sigma[-1, ]),
                                          "logit", 1.0, S)$cpu())
  testthat::expect_equal(with_na, dropped, tolerance = 1e-4)
})

testthat::test_that("gradients reach sigma and the coefficients", {
  skip_if_no_torch()
  set.seed(5)
  n = 20; sp = 3; df = 2
  tt = function(x) sjSDM:::sjsdm_tensor(x)
  mu = tt(matrix(rnorm(n * sp), n, sp))$requires_grad_(TRUE)
  sigma = tt(matrix(rnorm(sp * df, sd = 0.5), sp, df))$requires_grad_(TRUE)
  Y = tt(matrix(rbinom(n * sp, 1, 0.5), n, sp))
  l = sjSDM:::mvp_logLik(mu, Y, sigma, "probit", 1.70169, 20L)$sum()
  l$backward()
  testthat::expect_true(all(is.finite(as.numeric(sigma$grad$cpu()))))
  testthat::expect_true(all(is.finite(as.numeric(mu$grad$cpu()))))
  testthat::expect_gt(as.numeric(sigma$grad$abs()$sum()$cpu()), 0)
})

testthat::test_that("the optimizers minimise a quadratic", {
  skip_if_no_torch()
  # the five torch constructors the factories in backend_optim.R wire to; nothing is
  # hand-written any more, so this is the complete set the package can build
  gens = list(torch::optim_ignite_rmsprop, torch::optim_ignite_sgd, torch::optim_ignite_adam,
              torch::optim_ignite_adamw, torch::optim_ignite_adagrad)
  for (g in gens) {
    p = torch::torch_tensor(c(5, -5, 5), requires_grad = TRUE)
    o = g(list(p), lr = 0.1)
    start = as.numeric(p$detach()$pow(2)$sum())
    for (i in 1:60) { o$zero_grad(); l = p$pow(2)$sum(); l$backward(); o$step() }
    testthat::expect_lt(as.numeric(p$detach()$pow(2)$sum()), start)
  }
})

testthat::test_that("the optimizer factories build the optimizer they name", {
  skip_if_no_torch()
  # the config objects in sjSDM_configs.R reach the backend only through these, and four of
  # the five are the default of no fitted model, so they are exercised here directly. Each
  # factory is one line forwarding to one torch constructor, so the class is the assertion
  # that carries the block's title: it is what catches AdamW() wired to optim_ignite_adam.
  factories = list(
    optimizer_RMSprop = sjSDM:::optimizer_RMSprop(lr = 0.01, momentum = 0.1, centered = TRUE),
    optimizer_SGD     = sjSDM:::optimizer_SGD(lr = 0.01, momentum = 0.5, nesterov = TRUE),
    optimizer_Adam    = sjSDM:::optimizer_Adam(lr = 0.01, amsgrad = TRUE),
    optimizer_AdamW   = sjSDM:::optimizer_AdamW(lr = 0.01),
    optimizer_Adagrad = sjSDM:::optimizer_Adagrad(lr = 0.01)
  )
  built = c(optimizer_RMSprop = "optim_ignite_rmsprop", optimizer_SGD = "optim_ignite_sgd",
            optimizer_Adam = "optim_ignite_adam", optimizer_AdamW = "optim_ignite_adamw",
            optimizer_Adagrad = "optim_ignite_adagrad")
  for (nm in names(factories)) {
    p = torch::torch_tensor(c(5, -5, 5), requires_grad = TRUE)
    o = factories[[nm]](list(p))
    testthat::expect_true(inherits(o, built[[nm]]), label = nm)
    start = as.numeric(p$detach()$pow(2)$sum())
    for (i in 1:60) { o$zero_grad(); p$pow(2)$sum()$backward(); o$step() }
    testthat::expect_lt(as.numeric(p$detach()$pow(2)$sum()), start, label = nm)
  }
})

testthat::test_that("the deprecated optimizer names still build a working optimizer", {
  skip_if_no_torch()
  # Adamax, AdaBound, AccSGD and madgrad are exported CRAN API from 1.0.7 that no longer has
  # an implementation; they redirect with a message. Each used to be covered by its own
  # factory above, and this is what is left of that coverage: a redirect that mistranslates
  # an argument (AdaBound's amsbound -> Adam's amsgrad) fails only when a model is fitted.
  deprecated = c(Adamax = "optim_ignite_adam", AdaBound = "optim_ignite_adam",
                 AccSGD = "optim_ignite_sgd", madgrad = "optim_ignite_adam",
                 DiffGrad = "optim_ignite_adam")
  for (nm in names(deprecated)) {
    build = get(nm, envir = asNamespace("sjSDM"))
    testthat::expect_message(build(), "deprecated")
    cfg = suppressMessages(build())
    cfg$params$lr = 0.05
    p = torch::torch_tensor(c(5, -5, 5), requires_grad = TRUE)
    o = do.call(cfg$ff(), cfg$params)(list(p))
    testthat::expect_true(inherits(o, deprecated[[nm]]), label = nm)
    start = as.numeric(p$detach()$pow(2)$sum())
    for (i in 1:60) {
      o$zero_grad()
      p$pow(2)$sum()$backward()
      o$step()
    }
    testthat::expect_lt(as.numeric(p$detach()$pow(2)$sum()), start, label = nm)
  }
})

testthat::test_that("the optimizer branches that the defaults never reach", {
  skip_if_no_torch()
  variants = list(
    rmsprop_momentum = sjSDM:::optimizer_RMSprop(lr = 0.01, momentum = 0.9),
    rmsprop_centered = sjSDM:::optimizer_RMSprop(lr = 0.01, centered = TRUE),
    sgd_nesterov     = sjSDM:::optimizer_SGD(lr = 0.01, momentum = 0.9, nesterov = TRUE),
    sgd_dampening    = sjSDM:::optimizer_SGD(lr = 0.01, momentum = 0.9, dampening = 0.5),
    adam_amsgrad     = sjSDM:::optimizer_Adam(lr = 0.05, amsgrad = TRUE),
    adamw_decay      = sjSDM:::optimizer_AdamW(lr = 0.05, weight_decay = 0.1),
    adagrad_decay    = sjSDM:::optimizer_Adagrad(lr = 0.05, lr_decay = 0.01,
                                                 initial_accumulator_value = 0.1)
  )
  for (nm in names(variants)) {
    p = torch::torch_tensor(c(5, -5, 5), requires_grad = TRUE)
    o = variants[[nm]](list(p))
    start = as.numeric(p$detach()$pow(2)$sum())
    for (i in 1:60) { o$zero_grad(); p$pow(2)$sum()$backward(); o$step() }
    testthat::expect_lt(as.numeric(p$detach()$pow(2)$sum()), start, label = nm)
    testthat::expect_true(all(is.finite(as.numeric(p$detach()))), label = nm)
  }
})

testthat::test_that("zero column design matrices build", {
  skip_if_no_torch()
  m = sjSDM:::Model_sjSDM(blocks = list(list(input_shape = 0L, output_shape = 3L)),
                          loss = list(link = "probit", species = 3L, df = 2L),
                          optimizer = RMSprop(), seed = 1L)
  X = matrix(0, 10, 0)
  Y = matrix(rbinom(30, 1, 0.5), 10, 3)
  m$fit(X, Y, batch_size = 5L, epochs = 2L, sampling = 5L, verbose = FALSE)
  testthat::expect_true(is.finite(m$logLik(X, Y, batch_size = 5L, sampling = 5L)[[1]]))
})

testthat::test_that("every CUDA request is validated at the one funnel a rebuild goes through", {
  skip_if_no_torch()
  d = sjSDM:::sjsdm_device
  testthat::expect_equal(as.character(d("cpu")$type), "cpu")
  testthat::expect_equal(as.character(d("mps")$type), "mps")
  # "cuda:0" as a string and an out-of-range index used to reach libtorch and die there
  if (torch::cuda_is_available()) {
    for (alias in list("gpu", "cuda", "cuda:0", 0L)) {
      testthat::expect_equal(as.character(d(alias)$type), "cuda")
      testthat::expect_equal(d(alias)$index, 0)
    }
    n = torch::cuda_device_count()
    testthat::expect_error(d(n), "CUDA device")
    testthat::expect_error(d(paste0("cuda:", n)), "CUDA device")
  } else {
    for (alias in list("gpu", "cuda", "cuda:0", 0L, 3L))
      testthat::expect_warning(testthat::expect_equal(as.character(d(alias)$type), "cpu"))
  }

  dt = sjSDM:::sjsdm_dtype
  testthat::expect_true(dt("float32") == torch::torch_float32())
  testthat::expect_true(dt("float64") == torch::torch_float64())
  testthat::expect_true(dt(torch::torch_float64()) == torch::torch_float64())
  testthat::expect_true(dt("garbage") == torch::torch_float32())
})

testthat::test_that("the association matrix carries the unit diagonal", {
  skip_if_no_torch()
  set.seed(19)
  sp = 5L; df = 3L
  m = sjSDM:::Model_sjSDM(blocks = list(list(input_shape = 2L, output_shape = sp)),
                          loss = list(link = "probit", species = sp, df = df), seed = 1L)
  s = matrix(rnorm(sp * df, sd = 0.4), sp, df)
  sjSDM:::set_state(m$loss, list(sigma = s))
  testthat::expect_equal(m$get_sigma, s, tolerance = 1e-6)
  # sigma %*% t(sigma) + I, not sigma %*% t(sigma) -- getCov() reads this binding
  testthat::expect_equal(m$covariance, s %*% t(s) + diag(sp), tolerance = 1e-5)
})

testthat::test_that("sjsdm_tensor refuses arrays it would silently flatten", {
  skip_if_no_torch()
  testthat::expect_error(sjSDM:::sjsdm_tensor(array(1:8, c(2, 2, 2))), "dimension 3")
  testthat::expect_equal(sjSDM:::sjsdm_tensor(array(1:4, 4))$shape, c(4, 1))
  testthat::expect_equal(sjSDM:::sjsdm_tensor(1:4)$shape, c(4, 1))
})

testthat::test_that("integer responses with NA survive the conversion to torch", {
  skip_if_no_torch()
  Yi = matrix(c(1L, 0L, NA, 1L), 2, 2)
  t1 = sjSDM:::sjsdm_tensor(Yi)
  testthat::expect_equal(as.numeric(t1$isnan()$sum()$cpu()), 1)
  # the naive conversion turns NA_integer_ into the int minimum instead
  t2 = torch::torch_tensor(Yi, dtype = torch::torch_float32())
  testthat::expect_equal(as.numeric(t2$isnan()$sum()$cpu()), 0)
})

testthat::test_that("torch uses python negative indexing, so leave-one-out needs positive indices", {
  skip_if_no_torch()
  m = matrix(1:20, 5, 4, byrow = TRUE)
  t = sjSDM:::sjsdm_tensor(m)
  # R drops the row, torch counts from the end -- the trap that broke anova's
  # leave-one-out sigma before it was caught
  testthat::expect_equal(nrow(m[-2, , drop = FALSE]), 4L)
  testthat::expect_equal(t[-2, , drop = FALSE]$shape[1], 1L)
  # the form anova must use
  k = seq_len(5)[-2]
  testthat::expect_equal(t[k, , drop = FALSE]$shape[1], 4L)
  testthat::expect_equal(as.numeric(t[k, , drop = FALSE][, 1]), as.numeric(m[-2, 1]))
})

testthat::test_that("anova leave-one-out uses every species but one", {
  skip_if_no_torch()
  set.seed(1)
  X = matrix(rnorm(120 * 3), 120, 3)
  Y = matrix(rbinom(120 * 5, 1, 0.4), 120, 5)
  m = sjSDM(Y = Y, env = linear(X, ~.), iter = 10L, sampling = 50L, verbose = FALSE, seed = 3L)
  a = anova(m, samples = 200L, verbose = FALSE)
  sp = as.data.frame(a$species$R2_McFadden)
  testthat::expect_equal(nrow(sp), 5L)
  testthat::expect_true(all(is.finite(as.matrix(sp))))
  # with the negative-indexing bug the leave-one-out likelihood used one species'
  # sigma row broadcast over the rest, which drove Full R2 far outside [0, 1]
  testthat::expect_true(all(sp$Full > -1 & sp$Full < 1.5))
})
