#' @title sjSDM
#'
#' @description fast and accurate joint species model
#' 
#' @param X matrix of environmental predictors
#' @param Y matrix of species occurences/responses
#' @param formula formula object for predictors
#' @param df degree of freedom for covariance parametrization, if `NULL` df is set to `ncol(Y)/2`
#' @param l1_coefs strength of lasso regularization on environmental coefficients: `l1_coefs * sum(abs(coefs))`
#' @param l2_coefs strength of ridge regularization on environmental coefficients: `l2_coefs * sum(coefs^2)`
#' @param l1_cov strength of lasso regulIarization on covariances in species-species association matrix
#' @param l2_cov strength of ridge regularization on covariances in species-species association matrix
#' @param iter number of fitting iterations
#' @param step_size batch size for stochastic gradient descent, if `NULL` then step_size is set to: `step_size = 0.1*nrow(X)`
#' @param learning_rate learning rate for Adamax optimizer
#' @param parallel number of cpu cores for the data loader, only necessary for large datasets 
#' @param device which device to be used, "cpu" or "gpu"
#' @param dtype which data type, most GPU support only 32 bit floats.
#' 
#' @details The function fits a multivariate probit model via Monte-Carlo integration of the joint likelihood for all species. 
#' 
#' @note sjSDM depends on the anaconda python distribution and pytorch, which need to be installed before being able to use the sjSDM function. See \code{\link{install_sjSDM}} for details.  
#' 
#' @example /inst/examples/sjSDM-example.R
#' @seealso \code{\link{print.sjSDM}}, \code{\link{predict.sjSDM}}, \code{\link{coef.sjSDM}}, \code{\link{summary.sjSDM}}, \code{\link{getCov}}, \code{\link{simulate.sjSDM}}
#' @export
sjSDM = function(X = NULL, Y = NULL, formula = NULL, df = NULL, l1_coefs = 0.0, l2_coefs = 0.0, 
                 l1_cov = 0.0, l2_cov = 0.0, iter = 50L, step_size = NULL,learning_rate = 0.1, 
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
  
  out = list()
  
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
  out$cl = match.call()
  
  

  ### settings ##
  if(is.null(df)) df = as.integer(floor(ncol(Y) / 2))
  if(is.null(step_size)) step_size = as.integer(floor(nrow(X) * 0.1))
  else step_size = as.integer(step_size)

  #.onLoad()

  # if(any(sapply(out$names, function(n) stringr::str_detect(stringr::str_to_lower(n), "intercept")))) intercept = FALSE
  output = ncol(Y)
  input = ncol(X)
  
  out$get_model = function(){
    model = fa$Model_base(input, device = device, dtype = dtype)
    model$add_layer(fa$layers$Layer_dense(hidden = output,
                                          bias = FALSE,
                                          l1 = l1_coefs,
                                          l2 = l2_coefs,
                                          activation = NULL,
                                          device = device,
                                          dtype = dtype))
    model$build(df = df, l1 = l1_cov, l2 = l2_cov, optimizer = fa$optimizer_adamax(lr = learning_rate, weight_decay = 0.01))
    return(model)
  }
  model = out$get_model()
  
  time = system.time({model$fit(X, Y, batch_size = step_size, epochs = as.integer(iter), parallel = parallel)})[3]

  out$logLik = model$logLik(X, Y)
  out$model = model
  out$settings = list( df = df, l1_coefs = l1_coefs, l2_coefs = l2_coefs, 
                       l1_cov = l1_cov, l2_cov = l2_cov, iter = iter, 
                       step_size = step_size,learning_rate = learning_rate, 
                       parallel = parallel,device = device, dtype = dtype)
  out$time = time
  out$data = list(X = X, Y = Y)
  out$sessionInfo = utils::sessionInfo()
  out$weights = model$weights_numpy
  out$sigma = model$get_sigma_numpy()
  out$history = model$history
  torch$cuda$empty_cache()
  class(out) = "sjSDM"
  return(out)
}


#' Print a fitted sjSDM model
#' 
#' @param x a moel fitted by \code{\link{sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
print.sjSDM = function(x, ...) {
  cat("sjSDM model, see summary(model) for details \n")
}


#' Predict from a fitted sjSDM model
#' 
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param newdata newdata for predictions
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
predict.sjSDM = function(object, newdata = NULL, ...) {
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

#' Return coefficients from a fitted sjSDM model
#' 
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
coef.sjSDM = function(object, ...) {
  return(object$weights[[1]])
}

#' Return summary of a fitted sjSDM model
#' 
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#' @export
summary.sjSDM = function(object, ...) {

  out = list()

  coefs = coef.sjSDM(object)
  env = coefs[[1]]
  if(length(coefs) > 1) {
    env = rbind(t(coefs[[2]]), env)
  }
  env = data.frame(env)
  colnames(env) = paste0("sp", 1:ncol(env))
  rownames(env) = object$names
  # if(length(coefs) > 1) {
  #   rownames(env) = c("intercept", paste0("env", 1:(nrow(env)-1)))
  # } else {
  #   rownames(env) =paste0("env", 1:nrow(env))
  # }

  cat("LogLik: ", -object$logLik[[1]], "\n")
  cat("Deviance: ", 2*object$logLik[[1]], "\n\n")
  cat("Regularization loss: ", object$logLik[[2]], "\n\n")

  cov_m = object$sigma %*% t(object$sigma)
  cor_m = stats::cov2cor(cov_m)

  p_cor = round(cor_m, 3)
  p_cor[upper.tri(p_cor)] = 0.000
  colnames(p_cor) = paste0("sp", 1:ncol(p_cor))
  rownames(p_cor) = colnames(p_cor)

  cat("Species-species correlation matrix: \n\n")
  print(p_cor)
  cat("\n\n\n")

  cat("Coefficients (beta): \n\n")
  if(dim(env)[2] > 50) utils::head(env)
  else print(env)

  out$coefs = env
  out$logLik = object$logLik
  out$sigma = object$sigma
  out$cov = cov_m
  return(invisible(out))
}


#' Generates simulations from sjSDM model
#'
#' Simulate nsim responses from the fitted model
#'
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param nsim number of simulations
#' @param seed seed for random numer generator
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#'
#' @importFrom stats simulate
#' @export
simulate.sjSDM = function(object, nsim = 1, seed = NULL, ...) {
  object = checkModel(object)
  if(!is.null(seed)) {
    set.seed(seed)
    torch$cuda$manual_seed(seed)
    torch$manual_seed(seed)
  }
  preds = abind::abind(lapply(1:nsim, function(i) predict.sjSDM(object)), along = 0L)
  simulation = apply(preds, 2:3, function(p) stats::rbinom(nsim, 1L,p))
  return(simulation)
}



#' getCov
#'
#' get species-species assocation (covariance) matrix
#' @param object a model fitted by \code{\link{sjSDM}}
#' @export
getCov = function(object){
  if(!inherits(object, "sjSDM")) stop("Please provide sjSDM object")
  return(object$sigma %*% t(object$sigma))
}

#' Extract Log-Likelihood from a fitted sjSDM model
#'
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#'
#' @importFrom stats simulate
#' @export
logLik.sjSDM <- function(object, ...){
  return(object$logLik[[1]])
}







