#' @title sjSDM_DNN
#'
#' @description fast and accurate joint species deep neural network model
#' 
#' @param X matrix of environmental predictors
#' @param Y matrix of species occurences/responses
#' @param formula formula object for predictors
#' @param hidden hidden units in layers, length of hidden correspond to number of layers
#' @param activation activation functions for layer, must be of same length as hidden
#' @param df degree of freedom for covariance parametrization, if \code{NULL} df is set to \code{ncol(Y)/2}
#' @param l1_coefs strength of lasso regularization on weights: \code{l1_coefs * sum(abs(weights))}
#' @param l2_coefs strength of ridge regularization on weights: \code{l2_coefs * sum(weights^2)}`
#' @param l1_cov strength of lasso regulIarization on covariances in species-species association matrix
#' @param l2_cov strength of ridge regularization on covariances in species-species association matrix
#' @param iter number of fitting iterations
#' @param step_size batch size for stochastic gradient descent, if \code{NULL} then step_size is set to: \code{step_size = 0.1*nrow(X)}
#' @param learning_rate learning rate for Adamax optimizer
#' @param sampling number of sampling steps for Monte Carlo integreation
#' @param parallel number of cpu cores for the data loader, only necessary for large datasets 
#' @param device which device to be used, "cpu" or "gpu"
#' @param dtype which data type, most GPUs support only 32 bit floats.
#' 
#' @details The function fits a deep neural network. The last layer consist of multivariate probit link and the loss is calculated via Monte-Carlo integration of the joint likelihood for all species. 
#' 
#' @note sjSDM_DNN depends on the anaconda python distribution and pytorch, which need to be installed before being able to use the sjSDM function. See \code{\link{install_sjSDM}} for details.  
#' @seealso \code{\link{predict.sjSDM_DNN}},  \code{\link{summary.sjSDM_DNN}}, \code{\link{getWeights.sjSDM_DNN}}, \code{\link{setWeights.sjSDM_DNN}}, , \code{\link{plot.sjSDM_DNN}}
#' @example /inst/examples/sjSDM_DNN-example.R
#' @author Maximilian Pichler
#' @export
sjSDM_DNN = function(X = NULL, Y = NULL, formula = NULL,hidden = c(10L, 10L, 10L), activation = "relu",df = NULL, l1_coefs = 0.0, l2_coefs = 0.0, 
                 l1_cov = 0.0, l2_cov = 0.0, iter = 50L, step_size = NULL,learning_rate = 0.01, sampling = 100L,
                 parallel = 0L, device = "cpu", dtype = "float32") {
  stopifnot(
    is.matrix(X) || is.data.frame(X),
    is.matrix(Y),
    df > 0,
    l1_coefs >= 0,
    l2_coefs >= 0,
    l1_cov >= 0,
    l2_cov >= 0,
    iter >= 0,
    learning_rate >= 0
  )
  
  if(reticulate::py_is_null_xptr(fa)) .onLoad()
  
  out = list()
  
  if(length(hidden) > length(activation)) activation = rep(activation, length(hidden))
  
  if(is.numeric(device)) device = as.integer(device)
  
  if(device == "gpu") device = 0L
  
  if(is.data.frame(X)) {
    
    if(!is.null(formula)){
      mf = match.call()
      m = match("formula", names(mf))
      formula = stats::as.formula(mf[m]$formula)
      X = stats::model.matrix(formula, X)
    } else {
      formula = stats::as.formula("~.")
      X = stats::model.matrix(formula, X)
    }
    
  } else {
    
    if(!is.null(formula)) {
      mf = match.call()
      m = match("formula", names(mf))
      formula = stats::as.formula(mf[m]$formula)
      X = data.frame(X)
      X = stats::model.matrix(formula, X)
    } else {
      formula = stats::as.formula("~.")
      X = stats::model.matrix(formula,data.frame(X))
    }
  }
  
  out$formula = formula
  out$names = colnames(X)
  out$species = colnames(Y)
  out$cl = match.call()
  
  ### settings ##
  if(is.null(df)) df = as.integer(floor(ncol(Y) / 2))
  if(is.null(step_size)) step_size = as.integer(floor(nrow(X) * 0.1))
  else step_size = as.integer(step_size)
  
  output = ncol(Y)
  input = ncol(X)
  
  out$get_model = function(){
    model = sjSDM_model(input, device = device, dtype = dtype)
    for(i in 1:length(hidden)) model %>% layer_dense(units = as.integer(hidden[i]), activation = activation[i], use_bias = FALSE, kernel_l1 = l1_coefs, kernel_l2 = l2_coefs)
    model %>% layer_dense(units = output, use_bias = FALSE, kernel_l1 = l1_coefs, kernel_l2 = l2_coefs)
    model %>% compile(df = df, l1_cov = l1_cov, l2_cov = l2_cov, optimizer = optimizer_adamax(learning_rate = learning_rate, weight_decay = 0.01))
    return(model)
  }
  model = out$get_model()
  
  time = system.time({
    model %>% fit(X = X, Y = Y, batch_size = step_size, epochs = as.integer(iter), parallel = parallel)
    })[3]
  
  out$model = model
  out$settings = list( df = df, l1_coefs = l1_coefs, l2_coefs = l2_coefs, 
                       l1_cov = l1_cov, l2_cov = l2_cov, iter = iter, 
                       step_size = step_size,learning_rate = learning_rate, 
                       parallel = parallel,device = device, dtype = dtype)
  out$time = time
  out$data = list(X = X, Y = Y)
  out$sessionInfo = utils::sessionInfo()
  out$weights = getWeights(model)
  out$sigma = out$weights$sigma
  out$history = model$history
  torch$cuda$empty_cache()
  class(out) = "sjSDM_DNN"
  return(out)
}


#' Return summary of a fitted sjSDM DNN
#' 
#' @param object a model fitted by \code{\link{sjSDM_DNN}}
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#' @export
summary.sjSDM_DNN = function(object, ...) {
  summary(object$model)
}


#' Print a fitted sjSDM DNN
#' 
#' @param x a moel fitted by \code{\link{sjSDM_DNN}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
print.sjSDM_DNN = function(x, ...) {
  print(x$model)
}




#' Plot training history
#' 
#' @param x a model fitted by \code{\link{sjSDM_DNN}}
#' @param y unused argument
#' @param ... Additional arguments to pass to \code{plot()}
#' @export
plot.sjSDM_DNN = function(x, y, ...) {
  plot.sjSDM_model(x$model)
}


#' Predict from a fitted sjSDM DNN
#' 
#' @param object a model fitted by \code{\link{sjSDM_DNN}}
#' @param newdata newdata for predictions
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
predict.sjSDM_DNN = function(object, newdata = NULL, ...) {
  object = checkModel(object)
  if(is.null(newdata)) {
    return(object$model$predict(newdata = object$data$X))
  } else {
    if(is.data.frame(newdata)) {
      newdata = stats::model.matrix(object$formula, newdata)
    } else {
      newdata = stats::model.matrix(object$formula, data.frame(newdata))
    }
  }
  pred = object$model$predict(newdata = newdata, ...)
  return(pred)
}

