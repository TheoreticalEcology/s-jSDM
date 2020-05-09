#' Linear model of environmental response 
#' 
#' specify the model to be fitted
#' @param data matrix of environmental predictors
#' @param formula formula object for predictors
#' @param lambda lambda penality, strength of regularization: \eqn{\lambda * (lasso + ridge)}
#' @param alpha weighting between lasso and ridge: \eqn{(1 - \alpha) * |coefficients| + \alpha ||coefficients||^2}
#' 
#' @seealso \code{\link{DNN}}, \code{\link{sjSDM}}
#' @example /inst/examples/sjSDM-example.R
#' @export
linear = function(data = NULL, formula = NULL, lambda = 0.0, alpha = 0.5) {
  if(is.data.frame(data)) {
    
    if(!is.null(formula)){
      mf = match.call()
      m = match("formula", names(mf))
      formula = stats::as.formula(mf[m]$formula)
      X = stats::model.matrix(formula, data)
    } else {
      formula = stats::as.formula("~.")
      X = stats::model.matrix(formula, data)
    }
    
  } else {
    
    if(!is.null(formula)) {
      mf = match.call()
      m = match("formula", names(mf))
      formula = stats::as.formula(mf[m]$formula)
      X = data.frame(data)
      X = stats::model.matrix(formula, X)
    } else {
      formula = stats::as.formula("~.")
      X = stats::model.matrix(formula,data.frame(data))
    }
  }
  
  if(lambda == 0.0) {
    lambda = -99.9
  }
  
  out = list()
  out$formula = formula
  out$X = X
  out$l1_coef = (1-alpha)*lambda
  out$l2_coef = alpha*lambda
  class(out) = "linear"
  return(out)
}

#' Print a linear object
#' 
#' @param x object created by \code{\link{linear}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
print.linear = function(x, ...) {
  print(x$formula)
}


#' Non-linear nodel (deep neural network) of environmental responses
#' 
#' specify the model to be fitted
#' @param data matrix of environmental predictors
#' @param formula formula object for predictors
#' @param hidden hidden units in layers, length of hidden corresponds to number of layers
#' @param activation activation functions, can be of length one, or a vector of activation functions for each layer. Currently supported: tanh, relu, or sigmoid
#' @param lambda lambda penality, strength of regularization: \eqn{\lambda * (lasso + ridge)}
#' @param alpha weighting between lasso and ridge: \eqn{(1 - \alpha) * |weights| + \alpha ||weights||^2}
#' 
#' @seealso \code{\link{linear}}, \code{\link{sjSDM}}
#' @example /inst/examples/sjSDM-example.R
#' @export
DNN = function(data = NULL, formula = NULL, hidden = c(10L, 10L, 10L), activation = "relu",  lambda = 0.0, alpha = 0.5) {
  if(is.data.frame(data)) {
    
    if(!is.null(formula)){
      mf = match.call()
      m = match("formula", names(mf))
      formula = stats::as.formula(mf[m]$formula)
      X = stats::model.matrix(formula, data)
    } else {
      formula = stats::as.formula("~.")
      X = stats::model.matrix(formula, data)
    }
    
  } else {
    
    if(!is.null(formula)) {
      mf = match.call()
      m = match("formula", names(mf))
      formula = stats::as.formula(mf[m]$formula)
      X = data.frame(data)
      X = stats::model.matrix(formula, X)
    } else {
      formula = stats::as.formula("~.")
      X = stats::model.matrix(formula,data.frame(data))
    }
  }
  out = list()
  out$formula = formula
  out$X = X
  out$l1_coef = (1-alpha)*lambda
  out$l2_coef = alpha*lambda
  out$hidden = as.integer(hidden)
  if(length(hidden) != length(activation)) activation = rep(activation, length(hidden))
  out$activation = activation
  class(out) = "DNN"
  return(out)
}

#' Print a DNN object
#' 
#' @param x object created by \code{\link{DNN}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
print.DNN = function(x, ...) {
  print(x$formula)
  cat("\nLayers with n nodes: ", x$hidden)
}


#' biotic structure
#' 
#' define biotic (species-species) assocation (interaction) structur
#' @param df degree of freedom for covariance parametrization, if \code{NULL} df is set to \code{ncol(Y)/2}
#' @param lambda lambda penality, strength of regularization: \eqn{\lambda * (lasso + ridge)}
#' @param alpha weighting between lasso and ridge: \eqn{(1 - \alpha) * |covariances| + \alpha ||covariances||^2}
#' @param on_diag regularization on diagonals 
#' @param inverse regularization on the inverse covariance matrix
#' @param diag use diagonal marix with zeros (internal usage)
#' 
#' @seealso \code{\link{sjSDM}}
#' @example /inst/examples/sjSDM-example.R
#' @export
bioticStruct= function(df = NULL, lambda = 0.0, alpha = 0.5, on_diag = FALSE, inverse=FALSE, diag = FALSE) {
  out = list()
  out$l1_cov = (1-alpha)*lambda
  out$l2_cov = alpha*lambda
  out$inverse = inverse
  out$diag = diag
  if(!is.null(df)) out$df = as.integer(df)
  out$on_diag = on_diag
  class(out) = "bioticStruct"
  return(out)
}

#' Print a bioticStruct object
#' 
#' @param x object created by \code{\link{bioticStruct}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
print.bioticStruct = function(x, ...) {
  cat("df: ",x$df)
}




#' spatial random effects
#' 
#' define spatial random effects (random intercepts for sites)
#' @param re vector of factors or integers 
#' @seealso \code{\link{sjSDM}}
#' @example /inst/examples/sjSDM-example.R
#' @export
spatialRE = function(re = NULL) {
  out = list()
  re = as.factor(re)
  out$levels = levels(re)
  out$re = as.integer(re) - 1L
  class(out) = "spatialRE"
  return(out)
}

#' Print a spatialRE object
#' 
#' @param x object created by \code{\link{spatialRE}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
print.spatialRE = function(x, ...) {
}







