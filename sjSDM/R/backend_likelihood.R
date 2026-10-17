# Monte-Carlo multivariate probit likelihood, R torch implementation.
# Port of inst/python/sjSDM_py/dist_mvp.py and Model_sjSDM._build_loss_function.

# `response` and `predict_response` differ for probit on purpose: the likelihood uses
# sigmoid(1.70169 eta), predict() the exact normal CDF (BACKEND_rewrite_plan.md 6.4).
sjsdm_family = function(link) {
  f = list(link = link, bounded = link %in% c("probit", "logit", "linear"))

  f$response = switch(link,
                      probit = ,
                      logit  = function(eta) torch::torch_sigmoid(eta),
                      linear = function(eta) torch::torch_clamp(eta, 0.0, 1.0),
                      count  = ,
                      nbinom = function(eta) eta$exp(),
                      normal = function(eta) eta,
                      stop("unknown link: ", link))

  f$logprob = switch(link,
                     probit = ,
                     logit  = ,
                     linear = function(E, Y, theta)
                       E$log()$mul(Y)$add((1.0 - E)$log()$mul(1.0 - Y)),
                     count  = function(E, Y, theta)
                       E$log()$mul(Y)$sub(E)$sub(torch::torch_lgamma(Y + 1.0)),
                     nbinom = function(E, Y, theta) {
                       eps = 0.0001
                       r = 1.0 / (torch::nnf_softplus(theta) + eps)
                       p = torch::torch_clamp((1.0 - r / (r + E)) + eps, 0.0, 1.0 - eps)
                       torch::torch_lgamma(r + Y)$sub(torch::torch_lgamma(Y + 1.0))$sub(torch::torch_lgamma(r))$
                         add(r * torch::torch_log1p(-p))$add(Y * p$log())
                     },
                     normal = function(E, Y, theta) {
                       s = theta$exp()
                       E$sub(Y)$pow(2.0)$div(-2.0 * s$pow(2.0))$sub(s$log())$sub(0.5 * log(2 * pi))
                     })

  # prediction at latent zero
  f$predict_response = switch(link,
                              probit = function(mu, alpha) mu$div(sqrt(2))$erf()$add(1.0)$mul(0.5),
                              logit  = function(mu, alpha) torch::torch_sigmoid(mu$mul(alpha)),
                              linear = function(mu, alpha) torch::torch_clamp(mu, 0.0, 1.0),
                              count  = ,
                              nbinom = function(mu, alpha) mu$exp(),
                              normal = function(mu, alpha) mu)

  # E_z[link(mu + z sigma')] in closed form, v = rowSums(sigma^2) per species. NULL where
  # there is none and predict() has to sample.
  f$marginal = switch(link,
                      probit = function(mu, v)
                        mu$div(v$add(1.0)$sqrt())$div(sqrt(2))$erf()$add(1.0)$mul(0.5),
                      count  = ,
                      nbinom = function(mu, v) mu$add(v$div(2.0))$exp(),
                      normal = function(mu, v) mu,
                      NULL)
  f
}

# Log-density, score and observed information of log p(y | u) in the UNSCALED linear predictor
# u, i.e. before alpha. `ll` drops terms constant in u -- the mode-finder only needs differences.
# Only used to build the Laplace proposal; `linear` is clamped and has no usable curvature, so
# it returns NULL and the caller falls back to prior sampling.
sjsdm_derivs = function(link, alpha = 1.0) {
  ls = function(x) torch::nnf_softplus(x$neg())$neg()          # log sigmoid, stable
  switch(link,
         probit = ,
         logit  = function(u, Y, theta) {
           au = u$mul(alpha)
           p = torch::torch_sigmoid(au)
           list(ll = Y$mul(ls(au))$add((1 - Y)$mul(ls(au$neg()))),
                g = (Y - p)$mul(alpha), w = p$mul(1 - p)$mul(alpha^2))
         },
         count  = function(u, Y, theta) {
           lam = u$exp()
           list(ll = Y$mul(u)$sub(lam), g = Y - lam, w = lam)
         },
         nbinom = function(u, Y, theta) {
           r = (1.0 / (torch::nnf_softplus(theta) + 0.0001))$unsqueeze(1)
           m = u$exp()
           list(ll = Y$mul(u)$sub((r + Y)$mul((r + m)$log())),
                g = (Y - m)$mul(r)$div(r + m),
                w = m$mul(r)$mul(r + Y)$div((r + m)$pow(2)))
         },
         normal = function(u, Y, theta) {
           v = theta$exp()$pow(2)$unsqueeze(1)
           list(ll = (Y - u)$pow(2)$div(-2.0 * v), g = (Y - u)$div(v),
                w = torch::torch_ones_like(u)$div(v))
         },
         NULL)
}

# Newton on g(z) = sum_j log p(y_j | mu_j + (sigma z)_j) - ||z||^2 / 2, one mode per site.
# A positive definite Hessian gives a descent direction, not convergence: where a species is
# saturated its weight alpha^2 p(1-p) vanishes, the Hessian collapses towards I, and the step
# overshoots. Sites whose occurrences are far from what the fixed effects predict then diverge
# and the proposal is centred nowhere near the mass -- which the median ESS does not show. Hence
# a per-site backtracking line search on g itself, and a convergence check the caller can read.
# Convergence is judged on the Newton decrement sqrt(g' H^-1 g), not on |g|: the gradient
# scales with 1/residual variance, so a gaussian fit at small scale never reached a gradient
# tolerance although its mode was exact.
mvp_laplace_proposal = function(mu, Y, sigma, derivs, theta = NULL, steps = 50L, tol = 1e-4) {
  d = sigma$shape[2]
  eye = torch::torch_eye(d, dtype = mu$dtype, device = mu$device)
  obs = Y$isnan()$logical_not()$to(dtype = mu$dtype)
  Y0 = Y$masked_fill(Y$isnan(), 0.0)
  z = torch::torch_zeros(c(mu$shape[1], d), dtype = mu$dtype, device = mu$device)

  parts = function(zz) derivs(torch::torch_matmul(zz, sigma$t())$add(mu), Y0, theta)
  # sigma' diag(w_i) sigma per site, as two matmuls. A three-operand einsum picks a contraction
  # path through [sites, species, df, df] and segfaulted libtorch at df = 500.
  obj = function(pp, zz) pp$ll$mul(obs)$sum(dim = 2)$sub(zz$pow(2)$sum(dim = 2)$mul(0.5))
  grad = function(pp, zz) torch::torch_matmul(pp$g$mul(obs), sigma)$sub(zz)
  hess = function(pp) torch::torch_matmul(sigma$t()$unsqueeze(1),
                                          pp$w$mul(obs)$unsqueeze(3)$mul(sigma$unsqueeze(1)))$add(eye)

  pp = parts(z)
  f0 = obj(pp, z)
  newton = function(pp, zz) {
    g = grad(pp, zz)
    step = torch::torch_cholesky_solve(g$unsqueeze(3), torch::linalg_cholesky(hess(pp)))$squeeze(3)
    list(step = step, dec = g$mul(step)$sum(dim = 2)$clamp(min = 0)$sqrt())
  }
  for (i in seq_len(steps)) {
    nw = newton(pp, z)
    if (as.numeric(nw$dec$max()$cpu()) < tol) break
    step = nw$step
    t = torch::torch_ones_like(f0)
    for (bt in 1:25) {
      zn = z$add(step$mul(t$unsqueeze(2)))
      pn = parts(zn)
      fn = obj(pn, zn)
      bad = fn$isfinite()$logical_not()$logical_or(fn < f0)
      if (!as.logical(bad$any()$cpu())) break
      t = torch::torch_where(bad, t$mul(0.5), t)
    }
    z = z$add(step$mul(t$unsqueeze(2)))
    pp = parts(z)
    f0 = obj(pp, z)
  }
  pp = parts(z)
  list(mode = z, L = torch::linalg_cholesky(hess(pp)),
       maxdec = as.numeric(newton(pp, z)$dec$max()$cpu()))
}

# Draw from N(mode, H^-1) and return log p(z) - log q(z). z = mode + L^-T eps makes the
# quadratic form of the proposal exactly ||eps||^2, so no second solve is needed.
mvp_laplace_draw = function(prop, sampling) {
  eps = torch::torch_randn(c(as.integer(sampling), prop$mode$shape[1], prop$mode$shape[2]),
                           dtype = prop$mode$dtype, device = prop$mode$device)
  # one solve per site with `sampling` right-hand sides. Broadcasting L over the draw dimension
  # instead materialises [sampling, sites, df, df] and asked for 37 GiB at df = 500.
  z = torch::linalg_solve_triangular(prop$L$transpose(2, 3), eps$permute(c(2, 3, 1)),
                                     upper = TRUE)$permute(c(3, 1, 2))$add(prop$mode$unsqueeze(1))
  logdet = torch::torch_diagonal(prop$L, dim1 = 2, dim2 = 3)$log()$sum(dim = 2)
  list(z = z, lw = eps$pow(2)$sum(dim = 3)$mul(0.5)$
         sub(z$pow(2)$sum(dim = 3)$mul(0.5))$sub(logdet$unsqueeze(1)))
}

# (sum w)^2 / sum w^2 on the log scale; 1 means every draw counted, 1/S means one did
mvp_ess = function(lw) torch::torch_exp(torch::torch_logsumexp(lw, dim = 1)$mul(2)$
                                          sub(torch::torch_logsumexp(lw$mul(2), dim = 1)))

# keeps probabilities away from 0/1; meaningless on a rate, so only the bounded links get it
mvp_guard = function(E) E$mul(0.999999)$add(0.0000005)

# Draw the MC noise, [sampling, batch, df]
mvp_noise = function(sampling, batch, df, device, dtype = torch::torch_float32()) {
  torch::torch_randn(c(sampling, batch, df), device = device, dtype = dtype)
}

# eta = noise %*% t(sigma) + mu, scaled by alpha; [sampling, batch, species]
mvp_eta = function(mu, sigma, noise, alpha) {
  eta = torch::torch_matmul(noise, sigma$t())$add(mu)
  if (alpha != 1.0) eta = eta$mul(alpha)
  eta
}

mvp_response = function(eta, link, guard = TRUE) {
  f = sjsdm_family(link)
  E = f$response(eta)
  if (guard && f$bounded) mvp_guard(E) else E
}

#' Per-species log-probability of the MC draws
#'
#' Every link is elementwise in the species index, including the per-species `theta` and the
#' NA mask, so column j of the result depends on `sigma[j, ]` and `mu[, j]` alone. Dropping
#' species therefore drops columns and leaves the rest bit for bit unchanged, which is what
#' `anova()` leans on to get all leave-one-out likelihoods out of one forward pass.
#'
#' @param mu linear predictor, `(sites, species)` tensor
#' @param Y responses, `(sites, species)` tensor, may contain NaN
#' @param sigma `(species, df)` tensor
#' @param link one of probit, logit, linear, count, nbinom, normal, or a `sjsdm_family()`
#' @param alpha scaling of the linear predictor (1.70169 for probit)
#' @param sampling number of MC samples
#' @param theta dispersion/scale tensor for nbinom and normal
#' @param noise optional fixed `(sampling, sites, df)` tensor, for reproducibility
#' @param na_rm mask NaN responses out of the likelihood
#' @param legacy_guard reproduce the python probability guard on unbounded links
#' @return `(sampling, sites, species)` tensor of log-probabilities
#' @noRd
mvp_species_logprob = function(mu, Y, sigma, link = "probit", alpha = 1.0, sampling = 100L,
                               theta = NULL, noise = NULL, na_rm = TRUE, legacy_guard = FALSE) {
  f = if (is.list(link)) link else sjsdm_family(link)
  if (is.null(noise))
    noise = mvp_noise(sampling, mu$shape[1], sigma$shape[2], mu$device, mu$dtype)
  E = f$response(mvp_eta(mu, sigma, noise, alpha))
  if (f$bounded || legacy_guard) E = mvp_guard(E)

  na_mask = NULL
  if (na_rm) {
    isna = Y$isnan()
    if (as.logical(isna$any()$cpu())) {
      na_mask = isna$unsqueeze(1)$expand_as(E)
      Y = Y$masked_fill(isna, 0.0)
    }
  }
  lp = f$logprob(E, Y, theta)
  if (!is.null(na_mask)) lp = lp$masked_fill(na_mask, 0.0)
  lp
}

# -log( mean_s exp(logprob) ) over the sampling dimension. torch_logsumexp does the max-shift
# internally, so this is the same stabilised reduction the hand-rolled version did, in one
# kernel instead of seven
mvp_mc_nll = function(lp) torch::torch_logsumexp(lp, dim = 1)$neg()$add(log(lp$shape[1]))

#' Monte-Carlo joint negative log-likelihood
#'
#' @inheritParams mvp_species_logprob
#' @return `(sites)` tensor of negative log-likelihoods
#' @noRd
mvp_logLik = function(mu, Y, sigma, link = "probit", alpha = 1.0, sampling = 100L,
                      theta = NULL, noise = NULL, na_rm = TRUE, legacy_guard = FALSE) {
  mvp_mc_nll(mvp_species_logprob(mu, Y, sigma, link = link, alpha = alpha, sampling = sampling,
                                 theta = theta, noise = noise, na_rm = na_rm,
                                 legacy_guard = legacy_guard)$sum(dim = 3))
}

# The loss owns sigma and theta. They appear nowhere outside the likelihood, and as
# registered parameters they are collected, serialised and moved with the module instead of
# by a hand-kept list and four setters.
mvp_loss = torch::nn_module(
  "mvp_loss",
  initialize = function(link, species, df, alpha = 1.0, diag = FALSE, l1 = 0.0, l2 = 0.0,
                        reg_on_Cov = TRUE, reg_on_Diag = TRUE, inverse = FALSE, zero = FALSE,
                        dtype = torch::torch_float32(), device = NULL) {
    self$family = sjsdm_family(link)
    self$link = link
    self$alpha = alpha
    self$diag = diag
    self$species = as.integer(species)
    self$reg = list(l1 = l1, l2 = l2, on_Cov = reg_on_Cov, on_Diag = reg_on_Diag,
                    inverse = inverse)
    self$theta = NULL
    if (link == "nbinom")
      self$theta = torch::nn_parameter(torch::torch_ones(self$species, dtype = dtype,
                                                         device = device))
    if (link == "normal")
      self$theta = torch::nn_parameter(torch::torch_zeros(self$species, dtype = dtype,
                                                          device = device))
    if (zero) {
      # no latent term at all: anova()'s sub-models without associations. sigma = I would
      # leave a fixed unit latent variance, which only probit absorbs into its intercept.
      self$df = 1L
      self$register_buffer("sigma", torch::torch_zeros(c(self$species, 1L), dtype = dtype,
                                                       device = device))
    } else if (diag) {
      self$df = self$species
      # the independent-species model: fixed, so a buffer rather than a parameter
      self$register_buffer("sigma", torch::torch_eye(self$species, dtype = dtype,
                                                     device = device))
    } else {
      self$df = as.integer(df)
      b = sqrt(6.0 / (self$species + self$df))
      self$sigma = torch::nn_parameter(
        torch::torch_tensor(matrix(stats::runif(self$species * self$df, -b, b),
                                   self$species, self$df), dtype = dtype, device = device))
    }
  },

  # unreduced: one negative log-likelihood per site, the caller reduces
  forward = function(mu, Y, sampling = 100L, noise = NULL) {
    mvp_logLik(mu, Y, self$sigma, link = self$family, alpha = self$alpha,
               sampling = as.integer(sampling), theta = self$theta, noise = noise)
  },

  # z is not observed, so the marginal is E_z[link(mu + z sigma')], not link(mu); logit and
  # linear have no closed form and fall back to sampling.
  # `extra_v` is the per-site latent variance a random block contributes for a group the fit
  # never saw: there the group effect is integrated over its prior rather than plugged in.
  response = function(mu, link = TRUE, marginal = TRUE, sampling = 1000L, extra_v = NULL) {
    if (!link) return(mu)
    if (!marginal) {
      E = self$family$predict_response(mu, self$alpha)
      return(if (self$family$bounded) mvp_guard(E) else E)
    }
    if (is.null(self$family$marginal)) {
      noise = mvp_noise(as.integer(sampling), mu$shape[1], self$sigma$shape[2],
                        mu$device, mu$dtype)
      eta = mvp_eta(mu, self$sigma, noise, 1.0)
      if (!is.null(extra_v))
        eta = eta$add(torch::torch_randn_like(eta)$mul(extra_v$sqrt()))
      return(self$response(eta, TRUE, FALSE)$mean(dim = 1))
    }
    v = self$sigma$pow(2)$sum(dim = 2)
    if (!is.null(extra_v)) v = extra_v$add(v)
    E = self$family$marginal(mu, v)
    if (self$family$bounded) E = mvp_guard(E)
    E
  },

  penalty = function() {
    r = self$reg
    if (r$l1 <= 0.0 && r$l2 <= 0.0) return(list())
    s = self$sigma
    if (!r$on_Cov) {
      out = list()
      if (r$l1 > 0.0) out = c(out, list(s$abs()$sum()$mul(r$l1)))
      if (r$l2 > 0.0) out = c(out, list(s$pow(2.0)$sum()$mul(r$l2)))
      return(out)
    }
    d = if (r$on_Diag) 0L else 1L
    ss = s$matmul(s$t())
    if (r$inverse)
      ss = torch::linalg_inv(ss$add(torch::torch_eye(s$shape[1], dtype = s$dtype,
                                                     device = s$device)))
    v = NULL
    if (r$l1 > 0.0)
      v = ss$triu(d)$abs()$sum()$mul(r$l1)$add(ss$tril(-1)$abs()$sum()$mul(r$l1))
    if (r$l2 > 0.0) {
      q = ss$triu(d)$pow(2.0)$sum()$mul(r$l2)$add(ss$tril(-1)$pow(2.0)$sum()$mul(r$l2))
      v = if (is.null(v)) q else v$add(q)
    }
    list(v)
  },

  active = list(
    covariance = function() {
      s = self$sigma$detach()
      as.matrix(s$matmul(s$t())$add(torch::torch_eye(s$shape[1], dtype = s$dtype,
                                                     device = s$device))$cpu())
    },
    get_sigma = function() as.matrix(self$sigma$detach()$cpu()),
    get_theta = function() if (is.null(self$theta)) NULL else as.numeric(self$theta$detach()$cpu())
  )
)
