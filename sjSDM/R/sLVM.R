#' sLVM
#' @description Latent-variable model
#' 
#' @param Y species occurrences
#' @param X environmental (abiotic) covariates
#' @param formula formula for environment
#' @param lv number of latent variables
#' @param family supported distributions: \code{binomial(link=c("logit", "probit"))}, \code{poisson(link=c("log", "identity"))}
#' @param priors list of scale priors for beta, lv, and lf
#' @param posterior type of posterior distribution
#' @param iter number of optimization steps
#' @param step_size batch_size
#' @param lr learning_rate, can be also a list (for each parameter type)
#' @param device which device to be used, "cpu" or "gpu"
#' @param dtype which data type, most GPUs support only 32 bit floats.
#' 
#' @details The function fits the LVM (see Warton et al., 2015) with stochastic variational inference
#'
#'
#' sLVM depends on the anaconda python distribution and pytorch, which need to be installed before being able to use the sjSDM function. 
#' See \code{\link{install_sjSDM}}, \code{vignette("Dependencies", package = "sjSDM")}
#' 
#' @section Guide:
#' Guide is the distribution that is assumed for the posterior distribution
#' 
#' @references 
#' Warton, D. I., Blanchet, F. G., O’Hara, R. B., Ovaskainen, O., Taskinen, S., Walker, S. C., & Hui, F. K. (2015). So many variables: joint modeling in community ecology. Trends in Ecology & Evolution, 30(12), 766-779.
#'
#' @example /inst/examples/sLVM-example.R
#' @seealso \code{\link{print.sLVM}}, \code{\link{predict.sLVM}}, \code{\link{coef.sLVM}}, \code{\link{summary.sLVM}}, \code{\link{getCov}}, \code{\link{getLF}}, \code{\link{getCI}}, \code{\link{getLF}}
#' @author Maximilian Pichler
#' @export
sLVM = function(Y = NULL, X = NULL, formula = NULL, lv = 2L, family,
                priors = list(3.0, 1.0, 1.0), posterior = c("DiagonalNormal", "LaplaceApproximation", "LowRankMultivariateNormal", "Delta"),
                iter = 50L, step_size=20L, lr=list(0.1), device = "cpu", dtype = "float32") {
  
  check_module()
  
  stopifnot(
    family$family %in% c("binomial", "poisson"),
    family$link %in% c("log", "logit", "probit", "identity")
  )
  
  out = list()
  
  if(!inherits(lr, "list")) lr = as.list(rep(lr, 3))
  
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
  
  posterior = match.arg(posterior)
  out$posterior = posterior
  
  lv = as.integer(lv)
  
  out$get_model = function(){
    model = fa$Model_LVM(device=device, dtype=dtype)
    model$build(as.integer(c(nrow(X), ncol(X))), as.integer(ncol(Y)),  df = lv, guide=posterior, 
                scale_mu=priors[[1]], scale_lf=priors[[2]],scale_lv=priors[[3]], family = family$family, link = family$link)
    return(model)
  }
  model = out$get_model()
  
  if(!inherits(X, "matrix")) X = matrix(X, ncol=1L)
  
  time = system.time({model$fit(X, Y, lr =lr, batch_size = as.integer(step_size), epochs = as.integer(iter), parallel = 0L)})[3]
  out$model = model
  out$data = list(X = X, Y = Y)
  out$formula = formula
  out$names = colnames(X)
  out$family = family
  out$posterior=posterior
  out$prior=priors
  out$state = serialize_state(model)
  out$lf = model$lfs
  out$lv = model$lvs
  out$mu = model$weights
  out$logLik = model$getLogLik(X, Y)
  out$sessionInfo = utils::sessionInfo()
  out$time = time
  out$covariance = model$covariance
  out$posterior_samples = lapply(model$posterior_samples, function(p) p$data$cpu()$numpy())
  class(out) = "sLVM"
  return(out)
}
# com = simulate_SDM(env = 3L, species = 5L, sites = 400L)
# 
# m = sLVM(com$response, com$env_weights,
#          family = binomial("probit"), formula = ~0+.,posterior = "DiagonalNormal", lr = list(0.03), iter=100L, priors = list(2.0,1.0,1.0))
# plot(density(m$posterior_samples$mu[,3,2]))
# abline(v = quantile(m$posterior_samples$mu[,3,2], probs = c(0.025, 0.975)))
# abline(v = mean((m$posterior_samples$mu[,3,2])), col="red")
# abline(v = median((m$posterior_samples$mu[,3,2])), col="red")






#' Print a fitted sLVM model
#' 
#' @param x a model fitted by \code{\link{sLVM}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
print.sLVM = function(x, ...) {
  cat("sLVM model, see summary(model) for details \n")
}


#' Predict from a fitted sLVM model
#' 
#' @param object a model fitted by \code{\link{sLVM}}
#' @param newdata newdata for predictions
#' @param mean_field use means of parameter estimates or samples from their distribution
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
predict.sLVM = function(object, newdata = NULL, mean_field=FALSE, ...) {
  object = checkModel(object)
  
    
    if(is.null(newdata)) {
      return(object$model$predictPosterior(X = object$data$X))
    } else {
      if(is.data.frame(newdata)) {
        newdata = stats::model.matrix(object$formula, newdata)
      } else {
        newdata = stats::model.matrix(object$formula, data.frame(newdata))
      }
    }
    if(!inherits(newdata, "matrix")) newdata = matrix(newdata, ncol = 1L)
    pred = object$model$predict(newdata = newdata, mean_field=mean_field, ...)
    return(pred)
    
}





#' Return coefficients from a fitted sLVM model
#' 
#' @param object a model fitted by \code{\link{sLVM}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
coef.sLVM = function(object, ...) {
    return(object$mu)
}


#' Get Credible Intervals
#' @param object a model fitted by \code{\link{sLVM}}
#' @param lower lower quantile
#' @param upper upper quantile
#' @export
getCI = function(object, lower=0.025, upper=0.975){
  stopifnot(inherits(object, "sLVM"))
  apply(object$posterior_samples$mu,2:3, function(q) stats::quantile(q,probs=c(lower, upper)))
}


#' Get latent factors
#' @param object a model fitted by \code{\link{sLVM}}
#' @export
getLF = function(object) {
  stopifnot(inherits(object, "sLVM"))
  return(object$lf)
}

#' Get latent variables
#' @param object a model fitted by \code{\link{sLVM}}
#' @export
getLV = function(object) {
  stopifnot(inherits(object, "sLVM"))
  return(object$lv)
}


#' Return summary of a fitted sLVM model
#' 
#' @param object a model fitted by \code{\link{sLVM}}
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#' @export
summary.sLVM = function(object, ...) {
  
  out = list()
  
  cat("LogLik: ", -object$logLik, "\n")
  cat("Deviance: ", 2*object$logLik, "\n\n")
  
  cov_m = object$covariance
  cor_m = stats::cov2cor(cov_m)
  
  p_cor = round(cor_m, 3)
  p_cor[upper.tri(p_cor)] = 0.000
  colnames(p_cor) = paste0("sp", 1:ncol(p_cor))
  rownames(p_cor) = colnames(p_cor)
  
  if(dim(p_cor)[1] < 25) {
    cat("Species-species correlation matrix: \n\n")
    print(p_cor)
    cat("\n\n\n")
  }
  
  
  
    
    coefs = coef.sLVM(object)
    if(inherits(coefs, "list")) coefs = coefs[[1]]
    env = coefs
    
    env = data.frame(env)
    if(is.null(object$species)) colnames(env) = paste0("sp", 1:ncol(env))
    else colnames(env) = object$species
    rownames(env) = object$names
    
    CIs = getCI(object)
    
    within = ((round(as.vector(CIs[1,,]),3) > 0) ==  (round(as.vector(CIs[2,,]),3) > 0))
    ifelse(within,"*"," " )
    
    parse = function(cc) ifelse(cc > 0, paste0(" ",cc), as.character(cc))
    
    sp_env = apply(expand.grid( rownames(env), colnames(env)), 1, function(n) paste0("",n[2]," ", n[1]))
    ee = parse(round(as.vector(as.matrix(env)), 3))
    lc = parse(round(as.vector(CIs[1,,]),3))
    hc = parse(round(as.vector(CIs[2,,]),3))
    s = ifelse(within,"*"," " )
    cols = c(" ", "Estimate(mean)", "Lower CI", "Higher CI", "")
    
    
    lambda = function(x)gsub("\\s", " ", format(x, width=max(nchar(sp_env), nchar(cols))))
    
    coefmat = 
      cbind(
        paste0(lambda(sp_env), "\t"),
        paste0(lambda(ee), "\t"),
        paste0(lambda(lc), "\t"),
        paste0(lambda(hc), "\t"),
        paste0(lambda(s),"\t")
      )
    coefmat = rbind(paste0(lambda(cols), "\t"),coefmat)
  
  cat("Coefficients:\n")  
  cat(paste0(apply(coefmat, 1, function(k) paste0(k, collapse = "")), collapse = "\n"))
    
  cat("\n---\n")
  cat("'*': 0.0 not in CI")

  out$CI = CIs
  out$coefs = env
  out$logLik = object$logLik
  out$sigma = object$covariance
  out$cov = cov_m
  return(invisible(out))
}



#' Extract Log-Likelihood from a fitted sLVM model
#'
#' @param object a model fitted by \code{\link{sLVM}}
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#'
#' @importFrom stats simulate
#' @export
logLik.sLVM <- function(object, ...){
  return(object$logLik)
}



