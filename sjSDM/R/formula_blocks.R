# Blocks inside an environmental formula. A formula without NN() never reaches the parser --
# design() stays its only path -- so every model that was expressible before is bit-identical.

#' Neural network block inside an environmental formula
#'
#' @description
#' Marks a group of covariates as a non-linear block of the environmental predictor. The block
#' is a feed-forward network from its own covariates to the species, and its output is added to
#' the linear predictor, so the parametric terms beside it keep their interpretation.
#'
#' @param terms the covariates the block is a function of, written as formula terms
#'   (\code{a + b + c}, or \code{.} for every column of the data). Captured unevaluated.
#' @param hidden hidden units per layer, length of hidden corresponds to number of layers
#' @param activation activation functions, of length one or one per hidden layer. Currently
#'   supported: tanh, relu, leakyrelu, selu, or sigmoid
#' @param bias whether the hidden layers carry biases. The output layer never does, it would
#'   alias with the parametric intercept.
#' @param lambda lambda penalty on this block's weights, strength of regularization:
#'   \eqn{\lambda * (lasso + ridge)}
#' @param alpha weighting between lasso and ridge:
#'   \eqn{(1 - \alpha) * |weights| + \alpha ||weights||^2}
#' @param dropout probability of dropout rate
#'
#' @details
#' A covariate may not appear both in the parametric part of the formula and in an \code{NN()}
#' block: a linear output layer can represent a linear term exactly, so the likelihood is flat
#' along the split and only the ratio of the two penalties decides it. \code{~ temp + NN(.)} is
#' therefore an error, not a shorthand.
#'
#' The block has no interpretable coefficients, so \code{coef()} and \code{summary()} report the
#' parametric part only, and standard errors become conditional on the fitted block.
#'
#' @return
#' An S3 class of type 'NN', a specification object holding the unevaluated terms and the
#' architecture. It is evaluated by \code{\link{linear}} when the formula is parsed.
#'
#' @examples
#' \dontrun{
#' com = simulate_SDM(env = 3L, species = 5L, sites = 100L)
#' X = data.frame(com$env_weights)
#'
#' model = sjSDM(com$response,
#'               env = linear(X, ~ X1 + NN(X2 + X3, hidden = c(10L, 10L))),
#'               iter = 20L, verbose = FALSE)
#'
#' coef(model)  # the parametric part only
#'
#' ## switch the block off and refit
#' model_without = update(model, env_blocks = 1L, verbose = FALSE)
#' }
#'
#' @seealso \code{\link{linear}}, \code{\link{DNN}}, \code{\link{sjSDM}}
#' @import checkmate
#' @export
NN = function(terms, hidden = c(10L, 10L, 10L), activation = "selu", bias = TRUE,
              lambda = 0.0, alpha = 0.5, dropout = 0.0) {
  vars = substitute(terms)
  if (missing(terms))
    stop("NN() needs the covariates it is a function of, e.g. NN(a + b)", call. = FALSE)
  qassert(hidden, "X+[1,)")
  qassert(activation, "S+[1,)")
  qassert(bias, "B+")
  qassert(lambda, "R1[0,)")
  qassert(alpha, "R1[0,)")
  qassert(dropout, "R1[0,)")

  out = list(vars = vars, hidden = as.integer(hidden), activation = activation,
             bias = as.list(bias), dropout = if (dropout > 0.0) dropout else NULL,
             l1 = (1 - alpha) * lambda, l2 = alpha * lambda)
  class(out) = "NN"
  return(out)
}

#' Print an NN object
#'
#' @param x object created by \code{\link{NN}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#'
#' @return Invisible NN object
#'
#' @export
print.NN = function(x, ...) {
  cat("NN(", deparse(x$vars), ")\nLayers with n nodes: ", x$hidden, "\n")
  return(invisible(x))
}

#' Random-effect bar with an explicit rank
#'
#' @description
#' Wraps a random-effect bar so that the rank of its group-level latent factor can be set.
#' \code{re(x | g, df = 4)} is \code{(x | g)} with \code{df} latent dimensions instead of
#' the default.
#'
#' @param bar a random-effect bar, \code{lhs | group}, with \code{lme4} semantics.
#' @param df number of latent dimensions \eqn{d} of the group effect. Must be at least the
#'   number of design columns \eqn{q} the bar produces, otherwise the per-species
#'   \eqn{q \times q} covariance \eqn{\Lambda_j \Lambda_j'} is rank deficient and the
#'   intercept-slope correlation is forced to \eqn{\pm 1}. Defaults to \code{max(q, 5)}.
#' @param loading \code{"species"}, the default, gives every species its own loading
#'   \eqn{\Lambda} and so its own species covariance for the group effect. \code{"shared"}
#'   gives the bar one loading vector for all species, i.e. the classical mixed-model row
#'   effect in which a group shifts every species by the same amount.
#'
#' @details
#' \code{df} caps the rank of the \emph{cross-species} covariance of the group effect: at
#' \code{df = 1} every species' group effect is a multiple of one common group score. It does
#' not restrict the per-species covariance, which is unconstrained once \code{df >= q}.
#'
#' \code{loading = "shared"} has one scalar per group rather than a latent vector, so there is
#' no rank to choose and \code{df} alongside it is an error. The implied species covariance of
#' the group effect is \eqn{\lambda^2 \mathbf{1}\mathbf{1}'}, rank one by construction, and
#' with \eqn{q > 1} the bar's columns are likewise perfectly correlated: one group score
#' loaded onto all of them. \code{\link{getRE}} then reports \code{cov} without the species
#' dimension and \code{\link{getCor}}\code{(which = "re")} is not defined for the bar.
#'
#' The call is read out of the formula by the parser and is never evaluated, so \code{re()}
#' is syntax rather than a function to call.
#'
#' @return A list holding the bar, \code{df} and \code{loading}.
#'
#' @seealso \code{\link{linear}}, \code{\link{getRE}}, \code{\link{sjSDM}}
#' @export
re = function(bar, df = NULL, loading = "species")
  list(bar = substitute(bar), df = df, loading = loading)

# NN(hidden = 5, a + b) must resolve to the same block as NN(a + b, hidden = 5)
nn_inner = function(cl) match.call(NN, cl)$terms

# every NN() call of the formula, evaluated to its spec and carrying the term labels it covers
nn_calls = function(tt, data, env) {
  idx = attr(tt, "specials")$NN
  if (is.null(idx)) return(list())
  vars = attr(tt, "variables")
  labs = attr(tt, "term.labels")
  ord = attr(tt, "order")
  fac = attr(tt, "factors")
  lapply(idx, function(i) {
    cols = which(fac[i, ] > 0)
    if (any(ord[cols] > 1L))
      stop("NN() cannot appear inside an interaction: ", labs[cols[ord[cols] > 1L]][1],
           ". Put both covariates inside the block instead.", call. = FALSE)
    spec = eval(vars[[i + 1L]], envir = env)
    assert_class(spec, "NN")
    spec$label = labs[cols]
    spec$labels = attr(stats::terms(eval(call("~", nn_inner(vars[[i + 1L]])), env),
                                    data = data), "term.labels")
    spec
  })
}

# the formula the shared model frame is built from: every NN() call replaced by its own terms,
# so one model.frame() gives all blocks the same rows after NA handling
nn_substitute = function(e) {
  if (!is.call(e)) return(e)
  if (identical(e[[1L]], quote(NN))) return(nn_inner(e))
  for (k in seq_along(e)[-1L]) e[[k]] = nn_substitute(e[[k]])
  e
}

block_terms = function(tt, keep, intercept) {
  dropx = setdiff(seq_along(attr(tt, "term.labels")), keep)
  out = if (length(dropx)) stats::drop.terms(tt, dropx, keep.response = FALSE) else tt
  attr(out, "intercept") = as.integer(intercept)
  out
}

block_design = function(tt, mf) {
  X = stats::model.matrix(tt, mf)
  list(X = X, terms = tt, xlevels = stats::.getXlevels(tt, mf),
       intercept = "(Intercept)" %in% colnames(X))
}

# A network with a linear output layer represents a linear term exactly, so a covariate in two
# blocks leaves the likelihood flat along the split (PREDICTOR_plan.md 4.3). Checked on the
# term labels of the formula as written: the shared frame has already merged the duplicates.
check_block_overlap = function(ll) {
  v = unlist(lapply(ll, function(l)
    unique(unlist(lapply(l, function(s) all.vars(str2lang(s)))))))
  dup = unique(v[duplicated(v)])
  if (length(dup))
    stop("a covariate cannot be in both the parametric part and an NN() block, or in two ",
         "NN() blocks: ", paste(dup, collapse = ", "), call. = FALSE)
}

# reformulas reads at most one extra argument off a special and drops the whole bar without a
# word when it finds two, so re()'s arguments are lifted out of the call first and replaced by
# one index into them. Returns the rewritten expression and the arguments in call order.
re_lift = function(e, args) {
  if (!is.call(e)) return(list(e = e, args = args))
  if (identical(e[[1L]], quote(re))) {
    a = as.list(match.call(re, e))[-1L]
    args[[length(args) + 1L]] = a[names(a) != "bar"]
    return(list(e = call("re", a$bar, spec = length(args)), args = args))
  }
  for (k in seq_along(e)[-1L]) {
    r = re_lift(e[[k]], args)
    e[[k]] = r$e
    args = r$args
  }
  list(e = e, args = args)
}

# `||` into independent bars and `g/h` into `h:g` plus `g`, then the bars out of the fixed
# part, with re()'s arguments carried alongside. reformulas owns findbars/nobars since lme4
# 2.0-6; mkReTrms is deliberately not used (it reorders blocks and turns an NA level into
# zeros).
bar_specs = function(formula, bars) {
  env = environment(formula)
  lifted = re_lift(formula[[length(formula)]], list())
  s = reformulas::splitForm(
    reformulas::expandDoubleVerts(stats::as.formula(call("~", lifted$e), env = env)),
    specials = "re")
  # expandDoubleVerts strips re() off a `||` bar together with its arguments, and splitForm
  # leaves a nested `g/h` inside re() unexpanded where findbars splits it
  if (length(s$reTrmFormulas) != length(bars) ||
      sum(s$reTrmClasses == "re") != length(lifted$args))
    stop("re() cannot wrap a `||` or a nested `g/h` bar: the formula parser then either drops ",
         "its arguments or leaves the bar unexpanded. Write the bars out, e.g. ",
         "re(1 | g, df = 2) + re(0 + x | g, df = 2).", call. = FALSE)
  args = lapply(seq_along(s$reTrmFormulas), function(i) {
    if (s$reTrmClasses[i] != "re") return(list())
    lapply(lifted$args[[as.list(s$reTrmAddArgs[[i]])$spec]], eval, envir = env)
  })
  list(fixed = s$fixedFormula, bars = s$reTrmFormulas, args = args)
}

# the bar's left hand side as a design, built on the shared model frame so the rows match the
# other blocks. model.matrix() is given the terms object, never the formula, because `~ 1`
# has no variables and a model frame of it would have no rows either.
bar_design = function(lhs, mf, env) {
  tt = stats::terms(stats::as.formula(call("~", lhs), env = env), data = mf)
  X = stats::model.matrix(tt, mf)
  list(X = X, terms = tt, xlevels = stats::.getXlevels(tt, mf),
       intercept = "(Intercept)" %in% colnames(X))
}

# G_j = Lambda[j,,] Lambda[j,,]' is a Gram matrix of q vectors in R^d, so d < q forces the
# intercept-slope correlation to +-1 and fits a degenerate model in silence. A shared loading
# has one scalar per group instead of a latent vector, so d is 1 and not the user's to set.
re_blocks = function(bars, args, mf, env) {
  lapply(seq_along(bars), function(i) {
    b = bars[[i]]
    bd = bar_design(b[[2L]], mf, env)
    g = factor(eval(b[[3L]], mf, env))
    q = ncol(bd$X)
    a = args[[i]]
    loading = match.arg(a$loading, c("species", "shared"))
    if (loading == "shared" && !is.null(a$df))
      stop("df is not available with loading = \"shared\" in (", deparse(b), "): a shared ",
           "loading is one scalar per group, so there is no latent rank to choose.",
           call. = FALSE)
    if (!is.null(a$df)) qassert(a$df, "X1[1,)")
    d = if (loading == "shared") 1L else if (is.null(a$df)) max(q, 5L) else as.integer(a$df)
    if (loading == "species" && d < q)
      stop("df = ", d, " is below the ", q, " columns of (", deparse(b), "): the per-species ",
           "covariance would be forced to rank ", d, ", i.e. correlation +-1 between its ",
           "columns. Use df >= ", q, ".", call. = FALSE)
    c(bd, list(bar = b, group = deparse(b[[3L]]), index = as.integer(g), levels = levels(g),
               nlevels = nlevels(g), q = q, df = as.integer(d), loading = loading))
  })
}

# design() plus the NN and random blocks the formula asks for. The returned names are
# design()'s, so the parametric block stays on the config object itself and every existing
# reader keeps working.
design_blocks = function(formula, data) {
  bars = reformulas::findbars(formula)
  env = environment(formula)
  fixed = if (length(bars)) bar_specs(formula, bars) else list(fixed = formula, args = list())
  tt = stats::terms(fixed$fixed, specials = "NN", data = data)
  nn = nn_calls(tt, data, env)
  if (!length(nn) && !length(bars))
    return(c(design(formula, data), list(nn = list(), re = list())))

  par_labels = setdiff(attr(tt, "term.labels"), vapply(nn, function(b) b$label, ""))
  check_block_overlap(c(list(par_labels), lapply(nn, function(b) b$labels)))

  # one model frame for every block, so NA handling is shared; cito's mmn builds one per block
  # and can drop different rows per block without saying so
  rhs = nn_substitute(fixed$fixed[[length(fixed$fixed)]])
  mff = rhs
  for (b in bars) mff = call("+", mff, reformulas::subbars(b))
  mf = stats::model.frame(stats::as.formula(call("~", mff), env = env), data)
  ttf = stats::terms(stats::as.formula(call("~", rhs), env = env), data = data)
  tt = stats::terms(mf)
  labs = attr(tt, "term.labels")
  idx = lapply(nn, function(b) match(b$labels, labs))
  par = match(setdiff(attr(ttf, "term.labels"), unlist(lapply(nn, function(b) b$labels))), labs)
  if (anyNA(c(unlist(idx), par)))
    stop("a block asks for a term the shared model frame does not hold", call. = FALSE)

  tl = c(list(block_terms(tt, par, attr(ttf, "intercept"))),
         lapply(idx, function(k) block_terms(tt, k, 0L)))

  blocks = lapply(seq_along(nn), function(k)
    c(block_design(tl[[k + 1L]], mf),
      nn[[k]][c("hidden", "activation", "bias", "dropout", "l1", "l2")]))
  c(block_design(tl[[1L]], mf),
    list(nn = blocks, re = re_blocks(bars, fixed$args, mf, env)))
}

# every block's design, parametric first, in the order the net adds them. A random block
# travels as [Z | group index] so that batching and tensor conversion stay one code path.
config_X = function(config) {
  if (is.null(config)) return(NULL)
  c(list(config$X), lapply(config$nn, function(b) b$X),
    lapply(config$re, function(b) cbind(b$X, b$index)))
}

# the random blocks' designs for newdata. A level the fit never saw gets index 0, which the
# block reads as "draw from the prior": mean zero, variance Lambda Lambda'.
re_newdata = function(config, newdata) {
  if (!is.data.frame(newdata)) newdata = data.frame(newdata)
  lapply(config$re, function(b) {
    Z = stats::model.matrix(stats::delete.response(b$terms), newdata, xlev = b$xlevels)
    g = match(as.character(eval(b$bar[[3L]], newdata)), b$levels)
    cbind(Z, ifelse(is.na(g), 0L, g))
  })
}

# config -> the plain-R specification of each of its random blocks. `at` is the position of
# the block's design in the input list the net is called with.
net_random = function(config, out_shape, at0) {
  if (is.null(config)) return(list())
  lapply(seq_along(config$re), function(k) {
    b = config$re[[k]]
    list(species = out_shape, q = b$q, df = b$df, nlevels = b$nlevels, loading = b$loading,
         at = at0 + k)
  })
}

# config -> the plain-R architecture of each of its net blocks
net_blocks = function(config, out_shape, skip_intercept) {
  if (is.null(config)) return(NULL)
  dnn = inherits(config, "DNN")
  hidden = if (dnn) as.integer(config$hidden) else list()
  par = list(input_shape = ncol(config$X), output_shape = out_shape, hidden = hidden,
             activation = if (dnn) config$activation else "linear",
             bias = layer_bias(if (dnn) config$bias else list(FALSE), hidden, TRUE),
             dropout = config$dropout, l1 = config$l1_coef, l2 = config$l2_coef,
             intercept = skip_intercept && isTRUE(config$intercept))
  c(list(par), lapply(config$nn, function(b)
    list(input_shape = ncol(b$X), output_shape = out_shape, hidden = b$hidden,
         activation = b$activation, bias = layer_bias(b$bias, b$hidden, FALSE),
         dropout = b$dropout, l1 = b$l1, l2 = b$l2, intercept = FALSE)))
}

# "off" is removal from the sum. An NN block is dropped; the parametric block keeps its place,
# which summary() and se() read, and gets a zero-column design that emits exact zeros.
select_blocks = function(config, keep) {
  keep = as.integer(keep)
  if (!1L %in% keep) {
    config$formula = stats::as.formula("~0")
    config[c("X", "terms", "xlevels", "intercept")] = design(config$formula, config$data)
  }
  config$nn = config$nn[sort(intersect(keep - 1L, seq_along(config$nn)))]
  config
}
