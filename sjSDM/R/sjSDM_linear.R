#' @title sjSDM
#'
#' @description fast and accurate joint species model
#' 
#' @param Y matrix of species occurences/responses
#' @param env matrix of environmental predictors, object of type \code{\link{envLinear}} or \code{\link{envDNN}}
#' @param biotic defines biotic (species-species associations) structure, object of type \code{\link{bioticStruct}}
#' @param spatial defines spatial structure, object of type \code{\link{spatialXY}}
#' @param iter number of fitting iterations
#' @param step_size batch size for stochastic gradient descent, if \code{NULL} then step_size is set to: \code{step_size = 0.1*nrow(X)}
#' @param learning_rate learning rate for Adamax optimizer
#' @param se calculate standard errors for environmental coefficients
#' @param link probit or logit
#' @param sampling number of sampling steps for Monte Carlo integreation
#' @param parallel number of cpu cores for the data loader, only necessary for large datasets 
#' @param device which device to be used, "cpu" or "gpu"
#' @param dtype which data type, most GPUs support only 32 bit floats.
#' 
#' @details The function fits a multivariate probit model via Monte-Carlo integration (see Chen et al., 2018) of the joint likelihood for all species. See Pichler and Hartig, 2020 for benchmark results.
#' 
#' sjSDM depends on the anaconda python distribution and pytorch, which need to be installed before being able to use the sjSDM function. 
#' See \code{\link{install_sjSDM}}, \code{vignette("Dependencies", package = "sjSDM")}, or the section below for details.
#' 
#' @references 
#' Chen, D., Xue, Y., & Gomes, C. P. (2018). End-to-end learning for the deep multivariate probit model. arXiv preprint arXiv:1803.08591.
#' 
#' Pichler, M., and Hartig, F. (2020). A new method for faster and more accurate inference of species associations from novel community data. arXiv preprint arXiv:2003.05331.
#' 
#' 
#' @example /inst/examples/sjSDM-example.R
#' @seealso \code{\link{print.sjSDM}}, \code{\link{predict.sjSDM}}, \code{\link{coef.sjSDM}}, \code{\link{summary.sjSDM}}, \code{\link{getCov}}, \code{\link{simulate.sjSDM}}, \code{\link{getSe}}
#' @author Maximilian Pichler
#' @export
sjSDM = function(Y = NULL, 
                 env = NULL,
                 biotic = bioticStruct(),
                 spatial = NULL,
                 iter = 50L, 
                 step_size = NULL,
                 learning_rate = 0.01, 
                 se = FALSE, 
                 link = c("probit", "logit", "linear"),
                 sampling = 100L,
                 parallel = 0L, 
                 device = "cpu", 
                 dtype = "float32") {
  stopifnot(
    !is.null(Y),
    iter >= 0,
    learning_rate >= 0
  )
  
  check_modul()
  
  out = list()
  
  if(is.numeric(device)) device = as.integer(device)
  
  if(device == "gpu") device = 0L

  if(is.matrix(env) || is.data.frame(env)) env = envLinear(data = env)
  
  link = match.arg(link)
  

  out$formula = env$formula
  out$names = colnames(env$X)
  out$species = colnames(Y)
  out$cl = match.call()
  link = match.arg(link)

  ### settings ##
  if(is.null(biotic$df)) biotic$df = as.integer(floor(ncol(Y) / 2))
  if(is.null(step_size)) step_size = as.integer(floor(nrow(env$X) * 0.1))
  else step_size = as.integer(step_size)

  output = ncol(Y)
  input = ncol(env$X)
  
  out$get_model = function(){
    model = fa$Model_base(input, device = device, dtype = dtype)

    if(inherits(env, "envDNN")) {
      for(i in 1:length(env$hidden))
        model$add_layer(fa$layers$Layer_dense(hidden = env$hidden[i],
                                              bias = FALSE,
                                              l1 = env$l1_coef,
                                              l2 = env$l2_coef,
                                              activation = env$activation[i],
                                              device = device,
                                              dtype = dtype))
    } 
    model$add_layer(fa$layers$Layer_dense(hidden = output,
                                          bias = FALSE,
                                          l1 = env$l1_coef,
                                          l2 = env$l2_coef,
                                          activation = NULL,
                                          device = device,
                                          dtype = dtype))
    
    
    model$build(df = biotic$df, 
                l1 = biotic$l1_cov, 
                l2 = biotic$l2_cov, 
                reg_on_Diag = biotic$on_diag,
                optimizer = fa$optimizer_adamax(lr = learning_rate, weight_decay = 0.01), 
                link = link)
    return(model)
  }
  model = out$get_model()
  
  time = system.time({model$fit(env$X, Y, batch_size = step_size, epochs = as.integer(iter), parallel = parallel, sampling = as.integer(sampling))})[3]

  out$logLik = model$logLik(env$X, Y,batch_size = step_size,parallel = parallel)
  if(se && !inherits(env, "envDNN")) try({ out$se = t(abind::abind(model$se(env$X, Y, batch_size = step_size, parallel = parallel),along=0L)) })
  
  
  if(!inherits(env, "envLinear")) class(model) = c("sjSDM_model", class(model))
  
  out$model = model
  out$settings = list(biotic = biotic, env = env, spatial = spatial,iter = iter, 
                      step_size = step_size,learning_rate = learning_rate, 
                      parallel = parallel,device = device, dtype = dtype)
  out$time = time
  out$data = list(X = env$X, Y = Y)
  out$sessionInfo = utils::sessionInfo()
  out$weights = model$weights_numpy
  out$sigma = model$get_sigma_numpy()
  out$history = model$history
  torch$cuda$empty_cache()
  if(inherits(env, "envLinear")) class(out) = "sjSDM"
  else class(out) = "sjSDM_DNN"
  return(out)
}


#' Print a fitted sjSDM model
#' 
#' @param x a model fitted by \code{\link{sjSDM}}
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


#' Post hoc calculation of standard errors
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param step_size batch size for stochastic gradient descent
#' @param parallel number of cpu cores for the data loader, only necessary for large datasets 
#' @export
getSe = function(object, step_size = NULL, parallel = 0L){
  if(!inherits(object, "sjSDM")) stop("object must be of class sjSDM")
  if(is.null(step_size)) step_size = object$settings$step_size
  else step_size = as.integer(step_size)
  try({ object$se = t(abind::abind(object$model$se(object$data$X, object$data$Y, batch_size = step_size, parallel = parallel),along=0L)) })
  return(object)
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
  if(is.null(object$species)) colnames(env) = paste0("sp", 1:ncol(env))
  else colnames(env) = object$species
  rownames(env) = object$names

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
  
  
  # TO DO: p-value parsing:
  if(!is.null(object$se)) {
    out$z = object$weights[[1]][[1]] / object$se
    out$P = 2*stats::pnorm(abs(out$z),lower.tail = FALSE)
    out$se = object$se
    
    coefmat = cbind(
      as.vector(as.matrix(env)),
      as.vector(as.matrix(out$se)),
      as.vector(as.matrix(out$z)),
      as.vector(as.matrix(out$P))
      )
    colnames(coefmat) = c("Estimate", "Std.Err", "Z value", "Pr(>|z|)")
    rownames(coefmat) = apply(expand.grid( rownames(env), colnames(env)), 1, function(n) paste0(n[2]," ", n[1]))
    stats::printCoefmat(coefmat, signif.stars = getOption("show.signif.stars"), digits = 3)
    out$coefmat = coefmat
  } else {
  
  cat("Coefficients (beta): \n\n")
  if(dim(env)[2] > 50) utils::head(env)
  else print(env)
  }

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







