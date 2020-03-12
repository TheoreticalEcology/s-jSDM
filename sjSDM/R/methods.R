#' getCov
#'
#' get species-species assocation (covariance) matrix
#' @param object a model fitted by \code{\link{sjSDM}}, \code{\link{sjSDM_model}}, or \code{\link{sjSDM_DNN}}
#' @seealso \code{\link{sjSDM}}, \code{\link{sjSDM_model}}, \code{\link{sjSDM_DNN}}
#' @export
getCov = function(object) UseMethod("getCov")


#' @rdname getCov
#' @export
getCov.sjSDM = function(object){
  return(object$sigma %*% t(object$sigma))
}


#' @rdname getCov
#' @export
getCov.sjSDM_model = function(object){
  return(object$get_cov())
}


#' @rdname getCov
#' @export
getCov.sjSDM_DNN = function(object){
  return(object$sigma %*% t(object$sigma))
}





#' Get weights
#' 
#' return weights of each layer
#' @param object object of class \code{\link{sjSDM_DNN}} or of class \code{\link{sjSDM_model}}
#' @return 
#' \itemize{
#'  \item layers - list of layer weights
#'  \item sigma - weight to construct covariance matrix
#' }
#' @export
getWeights = function(object) UseMethod("getWeights")


#' @rdname getWeights
#' @export
getWeights.sjSDM_model = function(object) {
  return(list(layers = object$weights_numpy, sigma = object$sigma_numpy))
}


#' @rdname getWeights
#' @export
getWeights.sjSDM_DNN = function(object) {
  getWeights(object$model)
}


#' Set weights
#' 
#' set layer weights and sigma in \code{\link{sjSDM_model}} or \code{\link{sjSDM_DNN}} objects
#' @param object object of class \code{\link{sjSDM_model}} or \code{\link{sjSDM_DNN}} objects
#' @param weights list of layer weights and sigma, see \code{\link{getWeights}}
#' @export
setWeights = function(object, weights) UseMethod("setWeights")

#' @rdname setWeights
#' @export
setWeights.sjSDM_model = function(object, weights) {
  object$set_weights(weights$layers)
  object$set_sigma(weights$sigma)
  object$weights_numpy = weights$layers
  object$sigma_numpy = weights$sigma
  return(invisible(object))
}


#' @rdname setWeights
#' @export
setWeights.sjSDM_DNN = function(object, weights = NULL) {
  if(is.null(weights)) weights = object$weights
  setWeights(object$model, weights)
}