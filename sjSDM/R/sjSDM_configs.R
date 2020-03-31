#' linear
#' 
#' create linear env covariates
#' @param data matrix of environmental predictors
#' @param formula formula object for predictors
#' @param lambda lambda penality
#' @param alpha weighting between LASSO and ridge
#' @export
envLinear = function(data = NULL, formula = NULL, lambda = 0.0, alpha = 0.5) {
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
  class(out) = "envLinear"
  return(out)
}

#' env DNN
#' 
#' create deep neural network 
#' @param data matrix of environmental predictors
#' @param formula formula object for predictors
#' @param hidden hidden units in layers, length of hidden correspond to number of layers
#' @param activation activation functions for layer, must be of same length as hidden
#' @param lambda lambda penality
#' @param alpha weighting between LASSO and ridge
#' @export
envDNN = function(data = NULL, formula = NULL, hidden = c(10L, 10L, 10L), activation = "relu",  lambda = 0.0, alpha = 0.5) {
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
  class(out) = "envDNN"
  return(out)
}

#' biotic structure
#' 
#' define biotic interaction structur
#' @param df degree of freedom for covariance parametrization, if \code{NULL} df is set to \code{ncol(Y)/2}
#' @param lambda lambda penality
#' @param alpha weighting between LASSO and ridge
#' @param on_diag regularization on diagonals 
#' @export
bioticStruct= function(df = NULL, lambda = 0.0, alpha = 0.5, on_diag = TRUE) {
  out = list()
  out$l1_cov = (1-alpha)*lambda
  out$l2_cov = alpha*lambda
  if(!is.null(df)) out$df = as.integer(df)
  out$on_diag = on_diag
  class(out) = "bioticStruct"
  return(out)
}



#' spatial
#' 
#' define spatial structure, not yet supported
#' @export
spatialXY = function() {
  out = list()
  class(out) = "spatialXY"
  return(out)
}




