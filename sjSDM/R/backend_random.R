# Grouping-factor random effects, integrated out by a stochastic variational approximation.
# eta_ij += sum_k Z[i,k] (u_g' Lambda[j,k,]) with u_g ~ N(0, I_d) a priori and
# q_g = N(m_g, L_g L_g') the variational posterior. PREDICTOR_plan.md 3.3 and 6.2.

# L = tril(L_raw, -1) + diag(softplus(diag(L_raw))), so the diagonal is positive and
# log|L L'| stays finite; log(e - 1) makes softplus() exactly 1, i.e. L starts at I and the
# KL starts at exactly 0.
softplus_one = log(exp(1) - 1)

#' @noRd
random_block = torch::nn_module(
  "random_block",
  initialize = function(species, q, df, nlevels, loading = "species") {
    self$species = as.integer(species)
    self$q = as.integer(q)
    self$d = as.integer(df)
    self$nlevels = as.integer(nlevels)
    self$shared = identical(loading, "shared")
    b = sqrt(6.0 / (self$species + self$d))
    self$Lambda = torch::nn_parameter(torch::torch_tensor(
      if (self$shared) stats::runif(self$q, -b, b)
      else array(stats::runif(self$species * self$q * self$d, -b, b),
                 c(self$species, self$q, self$d))))
    self$m = torch::nn_parameter(torch::torch_zeros(c(self$nlevels, self$d)))
    self$L_raw = torch::nn_parameter(
      torch::torch_diag_embed(torch::torch_full(c(self$nlevels, self$d), softplus_one)))
  },

  # the classical mixed-model case: one scalar per group shifting every species identically,
  # which is Lambda restricted to a single [q] vector broadcast over species at d = 1.
  lambda = function() {
    if (!self$shared) return(self$Lambda)
    self$Lambda$reshape(c(1L, self$q, 1L))$expand(c(self$species, self$q, 1L))
  },

  L_mat = function() {
    dg = torch::torch_diagonal(self$L_raw, dim1 = 2, dim2 = 3)
    self$L_raw$tril(-1)$add(torch::torch_diag_embed(torch::nnf_softplus(dg)))
  },

  # u for every row of the batch. The sharing is here and nowhere else: two rows of the same
  # group read the same m_g and L_g, so the same eps gives them the same u. The variational
  # draw itself is per row, which is unbiased for the ELBO gradient and lower variance than
  # one draw per group (PREDICTOR_plan.md 6.2).
  u = function(x, sample = TRUE, eps = NULL) {
    idx = x$select(2, self$q + 1L)
    known = idx$gt(0.5)$unsqueeze(2)$to(dtype = x$dtype)
    i = idx$clamp(min = 1)$to(dtype = torch::torch_long())
    m = self$m$index_select(1, i)
    if (!sample) return(m$mul(known))
    if (is.null(eps))
      eps = torch::torch_randn(c(x$shape[1], self$d), dtype = x$dtype, device = x$device)
    L = self$L_mat()$index_select(1, i)
    m$add(torch::torch_bmm(L, eps$unsqueeze(3))$squeeze(3))$mul(known)
  },

  forward = function(x, sample = TRUE, eps = NULL) {
    u = self$u(x, sample, eps)
    Z = x$narrow(2, 1, self$q)
    zu = Z$unsqueeze(3)$mul(u$unsqueeze(2))$reshape(c(-1L, self$q * self$d))
    torch::torch_matmul(zu, self$lambda()$reshape(c(self$species, self$q * self$d))$t())
  },

  # KL(N(m, LL') || N(0, I_d)), summed over groups
  kl = function() {
    L = self$L_mat()
    dg = torch::torch_diagonal(L, dim1 = 2, dim2 = 3)
    L$pow(2.0)$sum()$add(self$m$pow(2.0)$sum())$sub(self$nlevels * self$d)$
      sub(dg$log()$sum()$mul(2.0))$mul(0.5)
  },

  # Var(sum_k Z[i,k] u' Lambda[j,k,]) under the prior, for rows whose group the fit never saw.
  # Zero elsewhere, because there the prediction conditions on m_g instead.
  prior_var = function(x) {
    unknown = x$select(2, self$q + 1L)$lt(0.5)$unsqueeze(2)$to(dtype = x$dtype)
    Z = x$narrow(2, 1, self$q)
    a = torch::torch_matmul(Z, self$lambda()$permute(c(2, 1, 3))$reshape(c(self$q, -1L)))
    a$reshape(c(-1L, self$species, self$d))$pow(2.0)$sum(dim = 3)$mul(unknown)
  }
)
