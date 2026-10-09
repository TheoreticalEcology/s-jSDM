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

#' Monte-Carlo joint negative log-likelihood
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
#' @return `(sites)` tensor of negative log-likelihoods
#' @noRd
mvp_logLik = function(mu, Y, sigma, link = "probit", alpha = 1.0, sampling = 100L,
                      theta = NULL, noise = NULL, na_rm = TRUE, legacy_guard = FALSE) {
  f = if (is.list(link)) link else sjsdm_family(link)
  batch = mu$shape[1]
  if (is.null(noise)) noise = mvp_noise(sampling, batch, sigma$shape[2], mu$device, mu$dtype)
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
  lp = lp$sum(dim = 3)
  # -log( mean_s exp(logprob) ). torch_logsumexp does the max-shift internally, so this is
  # the same stabilised reduction the hand-rolled version did, in one kernel instead of seven
  torch::torch_logsumexp(lp, dim = 1)$neg()$add(log(lp$shape[1]))
}

# The loss owns sigma and theta. They appear nowhere outside the likelihood, and as
# registered parameters they are collected, serialised and moved with the module instead of
# by a hand-kept list and four setters.
mvp_loss = torch::nn_module(
  "mvp_loss",
  initialize = function(link, species, df, alpha = 1.0, diag = FALSE, l1 = 0.0, l2 = 0.0,
                        reg_on_Cov = TRUE, reg_on_Diag = TRUE, inverse = FALSE,
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
    if (diag) {
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
