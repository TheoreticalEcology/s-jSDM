#' Linear model of environmental response 
#' 
#' specify the model to be fitted
#' @param data matrix of environmental predictors
#' @param formula formula object for predictors
#' @param lambda lambda penalty, strength of regularization: \eqn{\lambda * (lasso + ridge)}
#' @param alpha weighting between lasso and ridge: \eqn{(1 - \alpha) * |coefficients| + \alpha ||coefficients||^2}
#' 
#' @return
#' An S3 class of type 'linear' including the following components:
#' 
#' \item{formula}{Model matrix formula}
#' \item{X}{Model matrix of covariates}
#' \item{data}{Raw data}
#' \item{l1_coef}{L1 regularization strength, can be -99 if \code{lambda = 0.0}}
#' \item{l2_coef}{L2 regularization strength, can be -99 if \code{lambda = 0.0}}
#' 
#' Implemented S3 methods include \code{\link{print.linear}}
#' 
#' @seealso \code{\link{DNN}}, \code{\link{sjSDM}}
#' @example /inst/examples/sjSDM-example.R
#' @import checkmate
#' @export
linear = function(data = NULL, formula = NULL, lambda = 0.0, alpha = 0.5) {
  
  assert(checkMatrix(data), checkDataFrame(data))
  qassert(lambda, "R1[0,)")
  qassert(alpha, "R1[0,)")
  
  if(is.data.frame(data)) {
    
    if(!is.null(formula)){
      mf = match.call()
      m = match("formula", names(mf))
      if(inherits(mf[3]$formula, "name")) mf[3]$formula = eval(mf[3]$formula, envir = parent.env(environment()))
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
      if(inherits(mf[3]$formula, "name")) mf[3]$formula = eval(mf[3]$formula, envir = parent.env(environment()))
      formula = stats::as.formula(mf[m]$formula)
      data = data.frame(data)
      X = stats::model.matrix(formula, data)
    } else {
      formula = stats::as.formula("~.")
      data = data.frame(data)
      X = stats::model.matrix(formula,data)
    }
  }
  
  if(lambda == 0.0) {
    lambda = -99.9
  }
  
  out = list()
  out$formula = formula
  out$X = X
  out$data = data
  out$l1_coef = (1-alpha)*lambda
  out$l2_coef = alpha*lambda
  class(out) = "linear"
  return(out)
}

#' Print a linear object
#' 
#' @param x object created by \code{\link{linear}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' 
#' @return Invisible formula object
#' 
#' @export
print.linear = function(x, ...) {
  print(x$formula)
  return(invisible(x$formula))
}


#' Non-linear model (deep neural network) of environmental responses
#' 
#' specify the model to be fitted
#' @param data matrix of environmental predictors
#' @param formula formula object for predictors
#' @param hidden hidden units in layers, length of hidden corresponds to number of layers
#' @param activation activation functions, can be of length one, or a vector of activation functions for each layer. Currently supported: tanh, relu, leakyrelu, selu, or sigmoid
#' @param bias whether use biases in the layers, can be of length one, or a vector (number of hidden layers including (last layer) but not first layer (intercept in first layer is specified by formula)) of logicals for each layer.
#' @param lambda lambda penalty, strength of regularization: \eqn{\lambda * (lasso + ridge)}
#' @param alpha weighting between lasso and ridge: \eqn{(1 - \alpha) * |weights| + \alpha ||weights||^2}
#' @param dropout probability of dropout rate 
#' 
#' @return
#' An S3 class of type 'DNN' including the following components:
#' 
#' \item{formula}{Model matrix formula}
#' \item{X}{Model matrix of covariates}
#' \item{data}{Raw data}
#' \item{l1_coef}{L1 regularization strength, can be -99 if \code{lambda = 0.0}}
#' \item{l2_coef}{L2 regularization strength, can be -99 if \code{lambda = 0.0}}
#' \item{hidden}{Integer vector of hidden neurons in the deep neural network. Length of vector corresponds to the number of hidden layers.}
#' \item{activation}{Character vector of activation functions.}
#' \item{bias}{Logical vector whether to use bias or not in each hidden layer.}
#' 
#' Implemented S3 methods include \code{\link{print.DNN}}
#' 
#' @seealso \code{\link{linear}}, \code{\link{sjSDM}}
#' @example /inst/examples/sjSDM-example.R
#' @import checkmate
#' @export
DNN = function(data = NULL, formula = NULL, hidden = c(10L, 10L, 10L), activation = "relu", bias = TRUE, lambda = 0.0, alpha = 0.5, dropout = 0.0) {
  
  assert(checkMatrix(data), checkDataFrame(data))
  qassert(hidden, "X+[1,)")
  qassert(activation, "S+[1,)")
  qassert(bias, "B+")
  qassert(lambda, "R1[0,)")
  qassert(alpha, "R1[0,)")
  qassert(dropout, "R1[0,)")
  
  
  if(is.data.frame(data)) {
    
    if(!is.null(formula)){
      mf = match.call()
      m = match("formula", names(mf))
      if(inherits(mf[3]$formula, "name")) mf[3]$formula = eval(mf[3]$formula, envir = parent.env(environment()))
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
      if(inherits(mf[3]$formula, "name")) mf[3]$formula = eval(mf[3]$formula, envir = parent.env(environment()))
      formula = stats::as.formula(mf[m]$formula)
      data = data.frame(data)
      X = stats::model.matrix(formula, data)
    } else {
      formula = stats::as.formula("~.")
      data = data.frame(data)
      X = stats::model.matrix(formula, data)
    }
  }
  out = list()
  out$formula = formula
  out$X = X
  out$data = data
  out$l1_coef = (1-alpha)*lambda
  out$l2_coef = alpha*lambda
  out$hidden = as.integer(hidden)
  if(dropout > 0.0) out$dropout = dropout
  if(length(hidden) != length(activation)) activation = rep(activation, length(hidden))
  #if(length(hidden) > length(bias)) bias = rep(activation, length(hidden)+1)
  out$activation = activation
  out$bias = as.list(bias)
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
#' define biotic (species-species) association (interaction) structure
#' @param df degree of freedom for covariance parametrization, if \code{NULL} df is set to \code{ncol(Y)/2}
#' @param lambda lambda penalty, strength of regularization: \eqn{\lambda * (lasso + ridge)}
#' @param alpha weighting between lasso and ridge: \eqn{(1 - \alpha) * |covariances| + \alpha ||covariances||^2}
#' @param on_diag regularization on diagonals 
#' @param reg_on_Cov regularization on covariance matrix 
#' @param inverse regularization on the inverse covariance matrix
#' @param diag use diagonal matrix with zeros (internal usage)
#' 
#' @return
#' An S3 class of type 'bioticStruct' including the following components:
#' 
#' \item{l1_cov}{L1 regularization strength.}
#' \item{l2_cov}{L2 regularization strength.}
#' \item{inverse}{Logical, use inverse covariance matrix or not.}
#' \item{diag}{Logical, use diagonal matrix or not.}
#' \item{reg_on_Cov}{Logical, regularize covariance matrix or not.}
#' \item{on_diag}{Logical, regularize diagonals or not.}
#' 
#' Implemented S3 methods include \code{\link{print.bioticStruct}}
#' 
#' @seealso \code{\link{sjSDM}}
#' @example /inst/examples/sjSDM-example.R
#' @import checkmate
#' @export
bioticStruct= function(df = NULL, lambda = 0.0, alpha = 0.5, on_diag = FALSE,reg_on_Cov=TRUE, inverse=FALSE, diag = FALSE) {
  
  qassert(df, c("X1[1,)", "0"))
  qassert(lambda, "R1[0,)")
  qassert(alpha, "R1[0,)")
  qassert(on_diag, "B1")
  qassert(reg_on_Cov, "B1")
  qassert(inverse, "B1")
  qassert(diag, "B1")
  
  out = list()
  out$l1_cov = (1-alpha)*lambda
  out$l2_cov = alpha*lambda
  out$inverse = inverse
  out$diag = diag
  out$reg_on_Cov = reg_on_Cov
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



#' sjSDM control object
#' 
#' @param optimizer object of type \code{\link{RMSprop}}, \code{\link{Adamax}}, \code{\link{SGD}}, \code{\link{AccSGD}}, \code{\link{madgrad}}, or \code{\link{AdaBound}}
#' @param scheduler reduce lr on plateau scheduler or not (0 means no scheduler, > 0 number of epochs before reducing learning rate)
#' @param lr_reduce_factor factor to reduce learning rate in scheduler
#' @param early_stopping_training number of epochs without decrease in training loss before invoking early stopping (0 means no early stopping). 
#' @param mixed mixed (half-precision) training or not. Only recommended for GPUs > 2000 series
#' 
#' @return
#' List with the following fields:
#' 
#' \item{optimizer}{Function which returns an optimizer.}
#' \item{scheduler_boolean}{Logical, use scheduler or not.}
#' \item{scheduler_patience}{Integer, number of epochs to wait before applying plateau scheduler.}
#' \item{lr_reduce_factor}{Numerical, learning rate reduce factor.}
#' \item{mixed}{Logical, use mixed training or not.}
#' \item{early_stopping_training}{Numerical, early stopping after n epochs.}
#' 
#' @import checkmate
#' @export
sjSDMControl = function(optimizer = RMSprop(),
                        scheduler = 0,
                        lr_reduce_factor = 0.99,
                        early_stopping_training = 0,
                        mixed = FALSE) {
  
  qassert(optimizer, "L")
  qassert(scheduler, "X1[0,)")
  qassert(lr_reduce_factor, "R1(0,1)")
  qassert(early_stopping_training, "X1[0,)")
  qassert(mixed, "B1")
  
  control = list()
  control$optimizer = optimizer
  if(scheduler < 1) scheduler_boolean=FALSE
  else scheduler_boolean= TRUE
  control$scheduler_boolean = scheduler_boolean
  control$scheduler_patience = as.integer(scheduler)
  control$lr_reduce_factor = lr_reduce_factor
  control$mixed = mixed
  
  if(early_stopping_training == 0) early_stopping_training = -1.
  else early_stopping_training = as.integer(early_stopping_training)
  control$early_stopping_training = early_stopping_training
  return(control)
}


check_family = function(family){
  out = list()
  if(!family$family %in% c("binomial", "poisson", "gaussian")) stop(paste0(family$family, " ->  not supported"), call. = FALSE)
  
  if(family$family == "binomial"){
    if(!family$link %in% c("logit", "probit")){
      stop(paste0(family$link, " ->  not supported"), call. = FALSE)
    }
    out$link = family$link
  }
  if(family$family == "poisson"){
    if(!family$link %in% c("log")){
      stop(paste0(family$link, " ->  not supported"), call. = FALSE)
    }
    out$link = "count"
  }
  
  if(family$family == "gaussian"){
    if(!family$link %in% c("identity")){
      stop(paste0(family$link, " ->  not supported"), call. = FALSE)
    }
    out$link = "normal"
  }
  out$family = family
  return(out)
}




#' Adamax
#' 
#' Adamax optimizer, see Kingma and Ba, 2014
#' @param betas exponential decay rates
#' @param eps fuzz factor
#' @param weight_decay l2 penalty on weights
#' 
#' @return
#' Anonymous function that returns optimizer when called.
#' 
#' @references 
#' Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
#' @import checkmate
#' @export
Adamax = function(betas = c(0.9, 0.999), eps = 1e-08 , weight_decay = 0.002) {
  
  qassert(betas, "R2(0,1)")
  qassert(eps, "R1(0,)")
  qassert(weight_decay, "R1[0,)")
  
  out = list()
  out$params = list()
  out$params$betas = betas
  out$params$eps = eps
  out$params$weight_decay = weight_decay
  out$ff = function() pkg.env$fa$optimizer_adamax
  return(out)
}


#' RMSprop
#' 
#' RMSprop optimizer
#' @param alpha decay factor
#' @param eps fuzz factor
#' @param weight_decay l2 penalty on weights
#' @param momentum momentum
#' @param centered centered or not
#' 
#' @return
#' Anonymous function that returns optimizer when called.
#' @import checkmate
#' @export
RMSprop = function( alpha=0.99, eps=1e-8, weight_decay=0.01, momentum=0.1, centered=FALSE) {
  
  qassert(alpha, "R1(0,)")
  qassert(weight_decay, "R1[0,)")
  qassert(momentum, "R1[0,)")
  qassert(centered, "B1")
  
  out = list()
  out$params = list()
  out$params$alpha = alpha
  out$params$eps = eps
  out$params$weight_decay = weight_decay
  out$params$momentum = momentum
  out$params$centered = centered
  out$ff = function() pkg.env$fa$optimizer_RMSprop
  return(out)
}


#' SGD
#' 
#' stochastic gradient descent optimizer
#' @param momentum strength of momentum
#' @param dampening decay
#' @param weight_decay l2 penalty on weights
#' @param nesterov Nesterov momentum or not
#' @return
#' Anonymous function that returns optimizer when called.
#' @import checkmate
#' @export
SGD = function( momentum=0.5, dampening=0, weight_decay=0, nesterov=TRUE) {
  
  qassert(momentum, "R1(0,)")
  qassert(dampening, "R1[0,)")
  qassert(weight_decay, "R1[0,)")
  qassert(nesterov, "B1")
  
  out = list()
  out$params = list()
  out$params$momentum = momentum
  out$params$dampening = dampening
  out$params$weight_decay = weight_decay
  out$params$nesterov = nesterov
  out$ff = function() pkg.env$fa$optimizer_SGD
  return(out)
}


#' madgrad
#' 
#' stochastic gradient descent optimizer
#' @param momentum strength of momentum
#' @param weight_decay l2 penalty on weights
#' @param eps epsilon
#' @return
#' Anonymous function that returns optimizer when called.
#' @references 
#' Defazio, A., & Jelassi, S. (2021). Adaptivity without Compromise: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization. arXiv preprint arXiv:2101.11075.
#' @import checkmate
#' @export
madgrad = function(momentum=0.9, weight_decay=0, eps=1e-6) {
  
  qassert(momentum, "R1(0,)")
  qassert(weight_decay, "R1[0,)")
  qassert(eps, "R1(0,)")
  
  out = list()
  out$params = list()
  out$params$momentum = momentum
  out$params$weight_decay = weight_decay
  out$params$eps = eps
  out$ff = function() pkg.env$fa$optimizer_madgrad
  return(out)
}



#' AccSGD
#' 
#' accelerated stochastic gradient, see Kidambi et al., 2018 for details
#' @param kappa long step
#' @param xi advantage parameter
#' @param small_const small constant
#' @param weight_decay l2 penalty on weights
#' @return
#' Anonymous function that returns optimizer when called.
#' @references 
#' Kidambi, R., Netrapalli, P., Jain, P., & Kakade, S. (2018, February). On the insufficiency of existing momentum schemes for stochastic optimization. In 2018 Information Theory and Applications Workshop (ITA) (pp. 1-9). IEEE.
#' @import checkmate
#' @export
AccSGD = function(     kappa=1000.0,
                       xi=10.0,
                       small_const=0.7,
                       weight_decay=0) {
  
  qassert(kappa, "R1(0,)")
  qassert(xi, "R1[0,)")
  qassert(weight_decay, "R1[0,)")
  qassert(small_const, "R1[0,)")
  
  out = list()
  out$params = list(kappa=kappa,xi=xi,small_const=small_const,weight_decay=weight_decay)
  out$ff = function() pkg.env$fa$optimizer_AccSGD
  return(out)
}


#' AdaBound
#' 
#' adaptive gradient methods with dynamic bound of learning rate, see Luo et al., 2019 for details
#' @param betas betas
#' @param final_lr eps
#' @param gamma small_const
#' @param eps eps
#' @param weight_decay weight_decay
#' @param amsbound amsbound
#' @return
#' Anonymous function that returns optimizer when called.
#' @references 
#' Luo, L., Xiong, Y., Liu, Y., & Sun, X. (2019). Adaptive gradient methods with dynamic bound of learning rate. arXiv preprint arXiv:1902.09843.
#' @import checkmate
#' @export
AdaBound = function(    betas= c(0.9, 0.999),
                        final_lr = 0.1,
                        gamma=1e-3,
                        eps= 1e-8,
                        weight_decay=0,
                        amsbound=TRUE) {
  
  qassert(betas, "R2(0,1)")
  qassert(final_lr, "R1(0,)")
  qassert(gamma, "R1(0,)")
  qassert(eps, "R1(0,)")
  qassert(weight_decay, "R1[0,)")
  qassert(amsbound, "B1")
  
  out = list()
  out$params = list(betas=betas,final_lr=final_lr,gamma=gamma,eps= eps,weight_decay=weight_decay,amsbound=amsbound)
  out$ff = function() pkg.env$fa$optimizer_AdaBound
  return(out)
}


#' DiffGrad
#' @param betas betas
#' @param eps eps
#' @param weight_decay weight_decay
#' @return
#' Anonymous function that returns optimizer when called.
#' @import checkmate
DiffGrad = function(    betas=c(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=0) {
  
  qassert(betas, "R2(0,1)")
  qassert(eps, "R1(0,)")
  qassert(weight_decay, "R1[0,)")
  
  out = list()
  out$params = list(betas=betas,eps=eps,weight_decay=weight_decay)
  out$ff = function() pkg.env$fa$optimizer_DiffGrad
  return(out)
}