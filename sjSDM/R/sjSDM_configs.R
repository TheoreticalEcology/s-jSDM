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
      if(class(mf[3]$formula) == "name") mf[3]$formula = eval(mf[3]$formula, envir = parent.env(environment()))
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
      if(class(mf[3]$formula) == "name") mf[3]$formula = eval(mf[3]$formula, envir = parent.env(environment()))
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
#' @param bias whether use biases in the layers, can be of length one, or a vector (number of hidden layers + 1 (last layer)) of logicals for each layer.
#' @param lambda lambda penalty, strength of regularization: \eqn{\lambda * (lasso + ridge)}
#' @param alpha weighting between lasso and ridge: \eqn{(1 - \alpha) * |weights| + \alpha ||weights||^2}
#' @param dropout probability of dropout rate 
#' 
#' @seealso \code{\link{linear}}, \code{\link{sjSDM}}
#' @example /inst/examples/sjSDM-example.R
#' @export
DNN = function(data = NULL, formula = NULL, hidden = c(10L, 10L, 10L), activation = "relu", bias = TRUE, lambda = 0.0, alpha = 0.5, dropout = 0.0) {
  if(is.data.frame(data)) {
    
    if(!is.null(formula)){
      mf = match.call()
      m = match("formula", names(mf))
      if(class(mf[3]$formula) == "name") mf[3]$formula = eval(mf[3]$formula, envir = parent.env(environment()))
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
      if(class(mf[3]$formula) == "name") mf[3]$formula = eval(mf[3]$formula, envir = parent.env(environment()))
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
#' define biotic (species-species) assocation (interaction) structur
#' @param df degree of freedom for covariance parametrization, if \code{NULL} df is set to \code{ncol(Y)/2}
#' @param lambda lambda penality, strength of regularization: \eqn{\lambda * (lasso + ridge)}
#' @param alpha weighting between lasso and ridge: \eqn{(1 - \alpha) * |covariances| + \alpha ||covariances||^2}
#' @param on_diag regularization on diagonals 
#' @param reg_on_Cov regularization on covariance matrix 
#' @param inverse regularization on the inverse covariance matrix
#' @param diag use diagonal marix with zeros (internal usage)
#' 
#' @seealso \code{\link{sjSDM}}
#' @example /inst/examples/sjSDM-example.R
#' @export
bioticStruct= function(df = NULL, lambda = 0.0, alpha = 0.5, on_diag = FALSE,reg_on_Cov=TRUE, inverse=FALSE, diag = FALSE) {
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


#' sjSDM control object
#' 
#' @param optimizer object of type \code{\link{RMSprop}}, \code{\link{Adamax}}, \code{\link{SGD}}, \code{\link{AccSGD}}, or \code{\link{AdaBound}}
#' @param scheduler reduce lr on plateau scheduler or not (0 means no scheduler, > 0 number of epochs before reducing learning rate)
#' @param lr_reduce_factor factor to reduce learning rate in scheduler
#' @param early_stopping_training number of epochs without decrease in training loss before invoking early stopping (0 means no early stopping). 
#' 
#' @export
sjSDMControl = function(optimizer = RMSprop(),
                        scheduler = 0,
                        lr_reduce_factor = 0.99,
                        early_stopping_training = 0) {
  
  control = list()
  control$optimizer = optimizer
  if(scheduler < 1) scheduler_boolean=FALSE
  else scheduler_boolean= TRUE
  control$scheduler_boolean = scheduler_boolean
  control$scheduler_patience = as.integer(scheduler)
  control$lr_reduce_factor = lr_reduce_factor
  
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
#' @references 
#' Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
#' @export
Adamax = function(betas = c(0.9, 0.999), eps = 1e-08 , weight_decay = 0.002) {
  out = list()
  out$params = list()
  out$params$betas = betas
  out$params$eps = eps
  out$params$weight_decay = weight_decay
  out$ff = function() fa$optimizer_adamax
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
#' @export
RMSprop = function( alpha=0.99, eps=1e-8, weight_decay=0.01, momentum=0.1, centered=FALSE) {
  out = list()
  out$params = list()
  out$params$alpha = alpha
  out$params$eps = eps
  out$params$weight_decay = weight_decay
  out$params$momentum = momentum
  out$params$centered = centered
  out$ff = function() fa$optimizer_RMSprop
  return(out)
}


#' SGD
#' 
#' stochastic gradient descent optimizer
#' @param momentum strength of momentum
#' @param dampening decay
#' @param weight_decay l2 penalty on weights
#' @param nesterov Nesterov momentum or not
#' @export
SGD = function( momentum=0.5, dampening=0, weight_decay=0, nesterov=TRUE) {
  out = list()
  out$params = list()
  out$params$momentum = momentum
  out$params$dampening = dampening
  out$params$weight_decay = weight_decay
  out$params$nesterov = nesterov
  out$ff = function() fa$optimizer_SGD
  return(out)
}


#' AccSGD
#' 
#' accelerated stochastic gradient, see Kidambi et al., 2018 for details
#' @param kappa long step
#' @param xi advantage parameter
#' @param small_const small_const
#' @param weight_decay l2 penalty on weights
#' 
#' @references 
#' Kidambi, R., Netrapalli, P., Jain, P., & Kakade, S. (2018, February). On the insufficiency of existing momentum schemes for stochastic optimization. In 2018 Information Theory and Applications Workshop (ITA) (pp. 1-9). IEEE.
#' @export
AccSGD = function(     kappa=1000.0,
                       xi=10.0,
                       small_const=0.7,
                       weight_decay=0) {
  out = list()
  out$params = list(kappa=kappa,xi=xi,small_const=small_const,weight_decay=weight_decay)
  out$ff = function() fa$optimizer_AccSGD
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
#' 
#' @references 
#' Luo, L., Xiong, Y., Liu, Y., & Sun, X. (2019). Adaptive gradient methods with dynamic bound of learning rate. arXiv preprint arXiv:1902.09843.
#' 
#' @export
AdaBound = function(    betas= c(0.9, 0.999),
                        final_lr = 0.1,
                        gamma=1e-3,
                        eps= 1e-8,
                        weight_decay=0,
                        amsbound=TRUE) {
  out = list()
  out$params = list(betas=betas,final_lr=final_lr,gamma=gamma,eps= eps,weight_decay=weight_decay,amsbound=amsbound)
  out$ff = function() fa$optimizer_AdaBound
  return(out)
}


#' DiffGrad
#' @param betas betas
#' @param eps eps
#' @param weight_decay weight_decay
DiffGrad = function(    betas=c(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=0) {
  out = list()
  out$params = list(betas=betas,eps=eps,weight_decay=weight_decay)
  out$ff = function() fa$optimizer_DiffGrad
  return(out)
}