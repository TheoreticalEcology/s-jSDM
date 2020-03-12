#' getCov
#'
#' get species-species assocation (covariance) matrix
#' @param object a model fitted by \code{\link{sjSDM}} or of class \code{\link{sjSDM_model}}
#' @seealso \code{\link{sjSDM}}, \code{\link{sjSDM_model}}
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
