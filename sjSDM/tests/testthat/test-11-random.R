source("utils.R")

# Grouping-factor random effects. PREDICTOR_plan.md sections 3.3, 6.2 and 10.2.

re_data = function(n = 60L, G = 6L) {
  set.seed(21)
  data.frame(X1 = rnorm(n), X2 = rnorm(n), g = factor(rep(seq_len(G), length.out = n)),
             h = factor(rep(c("a", "b"), length.out = n)))
}

re_Y = function(n = 60L, sp = 4L) {
  set.seed(22)
  matrix(rbinom(n * sp, 1, 0.4), n, sp)
}

# linear() reads its formula out of match.call(), so a formula held in a variable has to go in
# through do.call() -- see the wart list in CLAUDE.md
env_re = function(f, X = re_data()) do.call(linear, list(data = X, formula = f))

fit_re = function(f, X = re_data(), Y = re_Y(), ...) {
  sjSDM(Y = Y, env = env_re(f, X), iter = 3L, sampling = 20L, verbose = FALSE,
        seed = 5L, device = "cpu", ...)
}

tt = function(x) sjSDM:::sjsdm_tensor(x)


testthat::test_that("bars parse with lme4 semantics", {
  skip_if_no_torch()
  b = env_re(~ X1 + (1 | g))$re
  testthat::expect_length(b, 1L)
  testthat::expect_equal(b[[1]]$q, 1L)
  testthat::expect_equal(colnames(b[[1]]$X), "(Intercept)")
  testthat::expect_equal(b[[1]]$nlevels, 6L)

  # (X1|g) is a correlated intercept AND slope
  b = env_re(~ X1 + (X1 | g))$re
  testthat::expect_length(b, 1L)
  testthat::expect_equal(colnames(b[[1]]$X), c("(Intercept)", "X1"))

  testthat::expect_equal(ncol(env_re(~ X1 + (0 + X1 | g))$re[[1]]$X), 1L)
  # || and / expand into independent bars
  testthat::expect_length(env_re(~ X1 + (X1 || g))$re, 2L)
  testthat::expect_length(env_re(~ 1 + (1 | g / h))$re, 2L)
  # the parametric block never sees the bar
  testthat::expect_equal(colnames(env_re(~ X1 + (1 | g))$X), c("(Intercept)", "X1"))
})


testthat::test_that("the rank condition d >= q is enforced", {
  skip_if_no_torch()
  testthat::expect_error(env_re(~ X1 + re(X1 | g, df = 1)), "rank")
  testthat::expect_equal(env_re(~ X1 + re(X1 | g, df = 2))$re[[1]]$df, 2L)
  testthat::expect_equal(env_re(~ X1 + re(1 | g, df = 3))$re[[1]]$df, 3L)
  # the default never violates it
  testthat::expect_true(env_re(~ (X1 | g))$re[[1]]$df >= 2L)
})


testthat::test_that("Lambda = 0, m = 0, L = I reduces exactly to the model without the block", {
  skip_if_no_torch()
  m = fit_re(~ X1 + (1 | g))
  b = m$model$net$random[[1]]
  torch::with_no_grad({
    b$Lambda$zero_()
    b$m$zero_()
    b$L_raw$copy_(torch::torch_diag_embed(
      torch::torch_full(c(b$nlevels, b$d), sjSDM:::softplus_one)))
  })
  inputs = lapply(sjSDM:::config_X(m$settings$env), tt)
  par = as.matrix(m$model$net$blocks[[1]](inputs[[1]])$cpu())
  testthat::expect_identical(as.matrix(m$model$net(inputs, TRUE)$cpu()), par)
  testthat::expect_identical(as.matrix(m$model$net(inputs, FALSE)$cpu()), par)
  testthat::expect_equal(as.numeric(b$kl()$cpu()), 0)
})


testthat::test_that("the block reproduces the hand written implementation under fixed noise", {
  skip_if_no_torch()
  set.seed(31)
  n = 12L
  sp = 3L
  q = 2L
  d = 3L
  G = 4L
  blk = sjSDM:::random_block(sp, q, d, G)
  Lam = array(rnorm(sp * q * d), c(sp, q, d))
  mu = matrix(rnorm(G * d), G, d)
  Lr = array(rnorm(G * d * d, 0, 0.3), c(G, d, d))
  torch::with_no_grad({
    blk$Lambda$copy_(torch::torch_tensor(Lam))
    blk$m$copy_(torch::torch_tensor(mu))
    blk$L_raw$copy_(torch::torch_tensor(Lr))
  })
  Z = matrix(rnorm(n * q), n, q)
  idx = sample.int(G, n, TRUE)
  eps = matrix(rnorm(n * d), n, d)
  got = as.matrix(blk(tt(cbind(Z, idx)), TRUE, tt(eps))$cpu())

  sp_f = function(x) log1p(exp(x))
  want = matrix(0, n, sp)
  for (i in seq_len(n)) {
    L = Lr[idx[i], , ]
    L[upper.tri(L, diag = TRUE)] = 0
    diag(L) = sp_f(diag(Lr[idx[i], , ]))
    u = mu[idx[i], ] + L %*% eps[i, ]
    for (j in seq_len(sp)) want[i, j] = sum(Z[i, ] * (matrix(Lam[j, , ], q, d) %*% u))
  }
  testthat::expect_equal(got, want, tolerance = 1e-5)

  # index 0 means a group the fit never saw: no mean, the prior variance instead
  z0 = as.matrix(blk(tt(cbind(Z, 0L)), FALSE)$cpu())
  testthat::expect_true(all(z0 == 0))
  a = sapply(seq_len(sp), function(j) rowSums((Z %*% matrix(Lam[j, , ], q, d))^2))
  testthat::expect_equal(as.matrix(blk$prior_var(tt(cbind(Z, 0L)))$cpu()), a, tolerance = 1e-5)
})


testthat::test_that("u is shared by the sites of a group and not across groups", {
  skip_if_no_torch()
  set.seed(32)
  sp = 3L
  d = 2L
  G = 3L
  blk = sjSDM:::random_block(sp, 1L, d, G)
  torch::with_no_grad({
    blk$m$copy_(torch::torch_tensor(matrix(rnorm(G * d), G, d)))
    blk$L_raw$copy_(torch::torch_tensor(array(rnorm(G * d * d, 0, 0.5), c(G, d, d))))
  })
  x = tt(cbind(rep(1, 4), c(1L, 1L, 2L, 2L)))
  eps = tt(matrix(rep(rnorm(d), each = 4), 4, d))
  u = as.matrix(blk$u(x, TRUE, eps)$cpu())
  testthat::expect_identical(u[1, ], u[2, ])
  testthat::expect_identical(u[3, ], u[4, ])
  testthat::expect_false(isTRUE(all.equal(u[1, ], u[3, ])))
  # and with the posterior means alone
  up = as.matrix(blk$u(x, FALSE)$cpu())
  testthat::expect_identical(up[1, ], up[2, ])
  testthat::expect_false(isTRUE(all.equal(up[1, ], up[3, ])))
})


testthat::test_that("the KL matches numerical integration at d = 1", {
  skip_if_no_torch()
  blk = sjSDM:::random_block(2L, 1L, 1L, 2L)
  mv = c(0.7, -1.3)
  sv = c(0.4, 2.1)
  torch::with_no_grad({
    blk$m$copy_(torch::torch_tensor(matrix(mv, 2L, 1L)))
    blk$L_raw$copy_(torch::torch_tensor(array(log(expm1(sv)), c(2L, 1L, 1L))))
  })
  num = sum(vapply(seq_along(mv), function(k)
    stats::integrate(function(x) stats::dnorm(x, mv[k], sv[k]) *
                       (stats::dnorm(x, mv[k], sv[k], log = TRUE) -
                          stats::dnorm(x, 0, 1, log = TRUE)), -Inf, Inf)$value, 1.0))
  testthat::expect_equal(as.numeric(blk$kl()$cpu()), num, tolerance = 1e-4)
})


testthat::test_that("the variational parameters sit in a weight_decay = 0 parameter group", {
  skip_if_no_torch()
  m = fit_re(~ X1 + (1 | g))
  g = m$model$net$parameter_groups()
  testthat::expect_setequal(names(g$variational), c("random.0.m", "random.0.L_raw"))
  testthat::expect_true(all(!grepl("^random\\.[0-9]+\\.(m|L_raw)$", names(g$fixed))))
  wd = vapply(m$model$optimizer$param_groups, function(p) p$weight_decay, 1.0)
  testthat::expect_equal(wd[2], 0)
  testthat::expect_true(wd[1] > 0)
})


testthat::test_that("a fitted model reports and restores its random effects", {
  skip_if_no_torch()
  m = fit_re(~ X1 + (X1 | g))
  r = getRE(m)
  testthat::expect_length(r, 1L)
  testthat::expect_equal(dim(r[[1]]$cov), c(4L, 2L, 2L))
  testthat::expect_equal(dim(r[[1]]$mean), c(6L, 2L, 4L))
  testthat::expect_equal(r[[1]]$group, "g")
  # Lambda Lambda' is symmetric positive semi definite for every species
  for (j in seq_len(dim(r[[1]]$cov)[1])) {
    testthat::expect_equal(r[[1]]$cov[j, , ], t(r[[1]]$cov[j, , ]))
    testthat::expect_true(min(eigen(r[[1]]$cov[j, , ], only.values = TRUE)$values) > -1e-6)
  }
  testthat::expect_null(getRE(fit_re(~ X1)))

  f = tempfile(fileext = ".RDS")
  saveRDS(m, f)
  on.exit(unlink(f))
  m2 = sjSDM:::checkModel(readRDS(f))
  testthat::expect_equal(getRE(m2)[[1]]$cov, r[[1]]$cov)
  testthat::expect_equal(predict(m2, marginal = FALSE), predict(m, marginal = FALSE))
})


testthat::test_that("anova() refuses a model with a random block", {
  skip_if_no_torch()
  testthat::expect_error(anova(fit_re(~ X1 + (1 | g)), samples = 20L, verbose = FALSE),
                         "random effects")
})


testthat::test_that("predict() falls back to the prior for an unseen group", {
  skip_if_no_torch()
  X = re_data()
  m = fit_re(~ X1 + (1 | g), X = X)
  nd = X
  nd$g = factor(rep("zzz", nrow(X)), levels = c(levels(X$g), "zzz"))
  testthat::expect_equal(predict(m, newdata = nd, type = "raw"),
                         predict(m, newdata = nd, type = "raw"))
  # no group effect in the mean, so the raw prediction is the parametric block alone
  par = as.matrix(m$model$net$blocks[[1]](tt(sjSDM:::config_X(m$settings$env)[[1]]))$cpu())
  testthat::expect_equal(predict(m, newdata = nd, type = "raw"), par, tolerance = 1e-5)
})


testthat::test_that("a shared loading shifts every species by the same amount", {
  skip_if_no_torch()
  b = env_re(~ X1 + re(1 | g, loading = "shared"))$re
  testthat::expect_equal(b[[1]]$loading, "shared")
  testthat::expect_equal(b[[1]]$df, 1L)
  testthat::expect_equal(env_re(~ X1 + (1 | g))$re[[1]]$loading, "species")

  m = fit_re(~ X1 + re(1 | g, loading = "shared"))
  blk = m$model$net$random[[1]]
  testthat::expect_equal(dim(blk$Lambda), 1L)
  x = sjSDM:::config_X(m$settings$env)[[2]]
  for (s in c(TRUE, FALSE)) {
    h = as.matrix(blk(tt(x), s)$cpu())
    testthat::expect_equal(max(apply(h, 1, function(z) diff(range(z)))), 0)
  }
  # the prior variance of an unseen group is likewise one number per row
  v = as.matrix(blk$prior_var(tt(cbind(x[, 1], 0L)))$cpu())
  testthat::expect_equal(max(apply(v, 1, function(z) diff(range(z)))), 0)
})


testthat::test_that("df and loading = 'shared' are exclusive, and re() rejects a || bar", {
  skip_if_no_torch()
  testthat::expect_error(env_re(~ X1 + re(1 | g, df = 3, loading = "shared")), "no latent rank")
  testthat::expect_error(env_re(~ X1 + re(X1 || g, df = 3)), "cannot wrap")
  testthat::expect_error(env_re(~ X1 + re(1 | g / h, df = 3)), "cannot wrap")
  testthat::expect_error(env_re(~ X1 + re(1 | g, loading = "every")), "species")
})


testthat::test_that("getRE() carries the loading, the Cholesky factor and the shapes", {
  skip_if_no_torch()
  m = fit_re(~ X1 + (X1 | g))
  r = getRE(m)[[1]]
  testthat::expect_equal(dim(r$lambda), c(4L, 2L, 5L))
  testthat::expect_equal(dim(r$L), c(6L, 5L, 5L))
  testthat::expect_identical(r$L, as.array(m$model$net$random[[1]]$L_mat()$detach()$cpu()))
  # cov is the Gram matrix of the loading, which is what L gets compared against
  for (j in 1:4)
    testthat::expect_equal(r$cov[j, , ], tcrossprod(matrix(r$lambda[j, , ], 2L, 5L)),
                           tolerance = 1e-6)

  rs = getRE(fit_re(~ X1 + re(1 | g, loading = "shared")))[[1]]
  testthat::expect_equal(rs$loading, "shared")
  testthat::expect_length(rs$lambda, 1L)
  testthat::expect_length(rs$cov, 1L)
  testthat::expect_equal(as.numeric(rs$cov), as.numeric(rs$lambda)^2)
  testthat::expect_equal(dim(rs$L), c(6L, 1L, 1L))
})


testthat::test_that("getCov(which = 're') is Lambda Lambda' and keeps the residual default", {
  skip_if_no_torch()
  m = fit_re(~ X1 + (1 | g))
  testthat::expect_identical(getCov(m), getCov(m, which = "residual"))
  testthat::expect_identical(getCor(m), getCor(m, which = "residual"))
  testthat::expect_error(getCov(fit_re(~ X1), which = "re"), "no random-effect bars")

  cv = getCov(m, which = "re")
  testthat::expect_named(cv, "g")
  lam = getRE(m)[[1]]$lambda
  testthat::expect_equal(unname(cv[[1]]), tcrossprod(matrix(lam, 4L, 5L)), tolerance = 1e-6)
  testthat::expect_equal(rownames(cv[[1]]), paste0("sp", 1:4))

  # q = 2: species j's block of the J q x J q matrix is getRE()$cov[j,,]
  m2 = fit_re(~ X1 + (X1 | g))
  cv2 = getCov(m2, which = "re")[[1]]
  testthat::expect_equal(dim(cv2), c(8L, 8L))
  testthat::expect_equal(rownames(cv2), paste0("sp", rep(1:4, 2), "_k", rep(1:2, each = 4)))
  r2 = getRE(m2)[[1]]
  for (j in 1:4)
    testthat::expect_equal(unname(cv2[j + c(0L, 4L), j + c(0L, 4L)]), r2$cov[j, , ],
                           tolerance = 1e-6)

  ms = fit_re(~ X1 + re(1 | g, loading = "shared"))
  testthat::expect_equal(getCov(ms, which = "re")[[1]], getRE(ms)[[1]]$cov)
  testthat::expect_error(getCor(ms, which = "re"), "loading = \"shared\"")
})


testthat::test_that("getCor(which = 're') warns on a rank deficient bar and guards zero variances", {
  skip_if_no_torch()
  # J = 4 species, q = 1, df = 5 >= J q: no deficiency
  m = fit_re(~ X1 + (1 | g))
  testthat::expect_silent(getCor(m, which = "re"))
  # df = 1 makes every species' group effect a multiple of one score
  m1 = fit_re(~ X1 + re(1 | g, df = 1))
  testthat::expect_warning(co <- getCor(m1, which = "re"), "singular")
  testthat::expect_equal(unname(abs(co[[1]])), matrix(1, 4L, 4L))

  lam = as.array(m1$model$net$random[[1]]$Lambda$detach()$cpu())
  lam[1, , ] = 0
  torch::with_no_grad(m1$model$net$random[[1]]$Lambda$copy_(torch::torch_tensor(lam)))
  co = suppressWarnings(getCor(m1, which = "re"))[[1]]
  testthat::expect_false(anyNA(co))
  testthat::expect_equal(unname(co[1, ]), rep(0, 4L))
})
