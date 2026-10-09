#' getCov
#'
#' get species-species association (covariance) matrix
#' @param object a model fitted by \code{\link{sjSDM}}, or \code{\link{sjSDM}} with \code{\link{DNN}} object
#' @param which \code{"residual"}, the default, for the residual association matrix
#'   \eqn{\sigma \sigma' + I}. \code{"re"} for the covariance a random-effect bar induces,
#'   \eqn{\Lambda \Lambda'}, one entry per bar.
#' @seealso \code{\link{sjSDM}},\code{\link{DNN}},\code{\link{getRE}}
#' 
#' @details
#' \code{which = "re"} returns a list named by the grouping variables. A bar with \eqn{q}
#' design columns gives the full \eqn{Jq \times Jq} covariance of the group effect, flattened
#' species-fast and named \code{<species>_k<column>}, so species \eqn{j}'s own
#' \eqn{q \times q} block is \code{\link{getRE}(object)[[b]]$cov[j,,]}; at \eqn{q = 1} it is
#' the plain \eqn{J \times J} matrix named by species. A bar with
#' \code{\link{re}(loading = "shared")} has no species dimension and returns the scalar
#' \eqn{\lambda^2}.
#' 
#' Unlike the residual covariance, \eqn{\Lambda \Lambda'} carries no \eqn{+ I}, so its rank
#' is at most \eqn{d} and it is singular whenever \eqn{d < Jq}. \code{\link{getCor}} warns
#' there: the group effects of the columns are linearly dependent, and at \eqn{d = 1} every
#' correlation is exactly \eqn{\pm 1}.
#' 
#' @return
#' 
#' Matrix of dimensions species by species corresponding to the covariance (occurrence) matrix,
#' or for \code{which = "re"} a named list with one covariance matrix per random-effect bar.
#' 
#' @export
getCov = function(object, which = c("residual", "re")) UseMethod("getCov")


#' @rdname getCov
#' @export
getCov.sjSDM = function(object, which = c("residual", "re")){
  if(match.arg(which) == "re") return(re_cov_list(object, FALSE))
  object = checkModel(object)
  return(force_r(object$model$covariance))
  #return(object$sigma %*% t(object$sigma))
}

#' getCor
#'
#' get species-species association correlation matrix
#' @param object a model fitted by \code{\link{sjSDM}}, or \code{\link{sjSDM}} with \code{\link{DNN}} object
#' @param which \code{"residual"}, the default, for the residual association matrix.
#'   \code{"re"} for the correlation a random-effect bar induces, one entry per bar; see
#'   \code{\link{getCov}} for the shape. Not defined for a bar with
#'   \code{\link{re}(loading = "shared")}, where it is 1 by construction.
#' @seealso \code{\link{sjSDM}},\code{\link{DNN}},\code{\link{getRE}}
#' 
#' @return
#' 
#' Matrix of dimensions species by species corresponding to the covariance (occurrence) matrix,
#' or for \code{which = "re"} a named list with one correlation matrix per random-effect bar.
#' 
#' @export
getCor = function(object, which = c("residual", "re")) UseMethod("getCor")


#' @rdname getCor
#' @export
getCor.sjSDM = function(object, which = c("residual", "re")){
  if(match.arg(which) == "re") return(re_cov_list(object, TRUE))
  object = checkModel(object)
  return(cov2cor(force_r(object$model$covariance)))
}

# cov2cor() divides by a zero scale in silence; a species whose loading is exactly zero has no
# group effect, and its row stays zero rather than turning into NaN.
re_cor = function(x) {
  s = sqrt(diag(x))
  s[s <= 0] = 1
  x / outer(s, s)
}

# Lambda Lambda' per bar. matrix() flattens the [J, q, d] loading species-fast, so the q x q
# diagonal block of species j is G_j = getRE()$cov[j,,].
re_cov_list = function(object, cor) {
  re = getRE(object)
  if(is.null(re)) stop("the model has no random-effect bars", call. = FALSE)
  out = lapply(re, function(r) {
    if(r$loading == "shared") {
      if(cor) stop("getCor(which = \"re\") is not defined for a bar with loading = \"shared\": ",
                   "every species carries the same group effect, so the correlation is 1 by ",
                   "construction. Use getCov(which = \"re\").", call. = FALSE)
      return(r$cov)
    }
    J = dim(r$lambda)[1]
    nm = if(is.null(object$species)) paste0("sp", seq_len(J)) else object$species
    if(r$q > 1L) nm = paste0(rep(nm, r$q), "_k", rep(seq_len(r$q), each = J))
    M = tcrossprod(matrix(r$lambda, J * r$q, r$df))
    dimnames(M) = list(nm, nm)
    if(!cor) return(M)
    if(r$df < J * r$q)
      warning("bar (", r$group, "): df = ", r$df, " is below J q = ", J * r$q, ", so ",
              "Lambda Lambda' has rank ", r$df, " and is singular -- the group effects of the ",
              "columns are linearly dependent, and at df = 1 every correlation is exactly +-1",
              call. = FALSE)
    re_cor(M)
  })
  return(stats::setNames(out, vapply(re, function(r) r$group, "")))
}


#' Get weights
#' 
#' return weights of each layer
#' @param object object of class \code{\link{sjSDM}} with \code{\link{DNN}}
#' @return 
#' \itemize{
#'  \item layers - list of layer weights
#'  \item sigma - weight to construct covariance matrix
#' }
#' @export
getWeights = function(object) UseMethod("getWeights")



#' @rdname getWeights
#' @export
getWeights.sjSDM= function(object) {
  return(list(env=force_r(object$model$env_weights), 
              spatial=force_r(object$model$spatial_weights), 
              sigma = force_r(object$model$get_sigma)))
}




#' Set weights
#' 
#' set layer weights and sigma in \code{\link{sjSDM}} with \code{\link{DNN}} object
#' @param object object of class  \code{\link{sjSDM}} with \code{\link{DNN}} object
#' @param weights list of layer weights:  \code{list(env=list(matrix(...)), spatial=list(matrix(...)), sigma=matrix(...))}, see \code{\link{getWeights}}
#' 
#' @return No return value, weights are changed in place. 
#' 
#' @export
setWeights = function(object, weights) UseMethod("setWeights")


#' @rdname setWeights
#' @export
setWeights.sjSDM= function(object, weights = NULL) {
  if(is.null(weights)) weights = list(env = object$weights, spatial = object$spatial_weights,
                                      sigma = object$sigma)
  
  if(!is.null(weights[[1]])) set_state(object$model$env, weights[[1]])
  
  if(inherits(object, "spatial") && length(weights) > 1 && !is.null(weights[[2]]))
    set_state(object$model$spatial, weights[[2]])
  
  if(length(weights) > 2 && !is.null(weights[[3]])) {
    sig = weights[[3]]
    if(inherits(sig, "list")) sig = unlist(sig)
    set_state(object$model$loss, list(sigma = sig))
  }
  # without this the new weights are undone by the next saveRDS/readRDS round trip
  sjsdm_state(object$model, object$state)
  invisible(NULL)
}

#' Get the random effects
#'
#' @description
#' Group-level latent factors of a model fitted with random-effect bars in the environmental
#' formula, one entry per bar.
#'
#' @param object a model fitted by \code{\link{sjSDM}}
#'
#' @details
#' \eqn{\Lambda_b} is identified only up to an orthogonal rotation of \eqn{R^{d_b}}, exactly
#' as \code{sigma} is, so the quantity to report is \code{cov}, the per-species covariance
#' \eqn{G_j = \Lambda_j \Lambda_j'} of the bar's design columns, and \code{mean}, the
#' posterior mean group effect \eqn{m_g' \Lambda_{jk}} on the linear predictor. \code{lambda}
#' and \code{L} are the raw ingredients, for a user who needs them: \code{L} is the posterior
#' Cholesky factor, the counterpart of \code{lme4}'s \code{condVar}, and \eqn{L_g L_g'} is
#' the posterior covariance of \eqn{u_g} that error bars on a group effect are built from.
#'
#' A bar with \code{\link{re}(loading = "shared")} has no species dimension: \code{lambda} is
#' a vector of length \eqn{q} and \code{cov} is \eqn{\lambda \lambda'}, a scalar for
#' \code{(1 | g)}, rather than the \code{species} by \eqn{q} by \eqn{q} array.
#'
#' @return A list with one element per bar, each holding \code{group} (the grouping variable),
#' \code{terms} and \code{levels}, \code{q} (the number of design columns of the bar),
#' \code{nlevels}, \code{df}, \code{loading}, \code{cov} (species by q by
#' q), \code{mean} (levels by q by species), \code{lambda} (species by q by df, the loading
#' \eqn{\Lambda}) and \code{L} (levels by df by df, the posterior Cholesky factor).
#' \code{NULL} for a model without random effects.
#'
#' @examples
#' \dontrun{
#' com = simulate_SDM(env = 3L, species = 5L, sites = 60L)
#' X = data.frame(com$env_weights)
#' X$plot = factor(rep(1:6, each = 10))
#'
#' model = sjSDM(com$response, env = linear(X, ~ X1 + re(1 | plot, df = 2)),
#'               iter = 10L, verbose = FALSE)
#'
#' rr = getRE(model)
#' dim(rr[[1]]$cov)   # species by q by q
#' dim(rr[[1]]$mean)  # groups by q by species
#'
#' getCov(model, which = "re")$plot
#' }
#'
#' @seealso \code{\link{sjSDM}}, \code{\link{re}}, \code{\link{getCov}}
#' @export
getRE = function(object) UseMethod("getRE")


#' @rdname getRE
#' @export
getRE.sjSDM = function(object) {
  re = object$settings$env$re
  if(!length(re)) return(NULL)
  object = checkModel(object)
  out = force_r(object$model$random_effects)
  for(k in seq_along(out))
    out[[k]][c("group", "levels", "terms")] =
      list(re[[k]]$group, re[[k]]$levels, colnames(re[[k]]$X))
  return(out)
}
