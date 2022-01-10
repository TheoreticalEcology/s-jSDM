#' getCov
#'
#' get species-species association (covariance) matrix
#' @param object a model fitted by \code{\link{sjSDM}}, or \code{\link{sjSDM}} with \code{\link{DNN}} object
#' @seealso \code{\link{sjSDM}},\code{\link{DNN}}
#' 
#' @return
#' 
#' Matrix of dimensions species by species corresponding to the covariance (occurrence) matrix.  
#' 
#' @export
getCov = function(object) UseMethod("getCov")


#' @rdname getCov
#' @export
getCov.sjSDM = function(object){
  object = checkModel(object)
  return(force_r(object$model$covariance))
  #return(object$sigma %*% t(object$sigma))
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
#' @param weights list of layer weights and sigma, see \code{\link{getWeights}}
#' 
#' @return No return value, weights are changed in place. 
#' 
#' @export
setWeights = function(object, weights) UseMethod("setWeights")


#' @rdname setWeights
#' @export
setWeights.sjSDM= function(object, weights = NULL) {
  if(is.null(weights)) weights = list(env = object$weights, spatial = object$spatial_weights)
  
  object$model$set_env_weights(w = weights[[1]])
  object$model$set_spatial_weights(w = weights[[2]])
}