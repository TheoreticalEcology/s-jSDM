#' sLVM
#' scalable LVM model
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
#' 
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
  }
  model = out$get_model()
  time = system.time({model$fit(reticulate::r_to_py(X)$copy(), 
                                reticulate::r_to_py(Y)$copy(), 
                                df = lv, guide=posterior, 
                                scale_mu=priors[[1]], scale_lf=priors[[2]],scale_lv=priors[[3]],
                                lr =lr, batch_size = as.integer(step_size), epochs = as.integer(iter), parallel = 0L)})[3]
  out$model = model
  out$data = list(X = X, Y = Y)
  out$formula = formula
  out$family = family
  out$posterior=posterior
  out$prior=priors
  out$state = serialize_state(model)
  out$lf = model$lfs
  out$lv = model$lvs
  out$mu = model$weights
  out$posterior_samples = lapply(model$posterior_samples, function(p) p$data$squeeze()$cpu()$numpy())
  class(out) = "LVM"
  return(out)
}
com = simulate_SDM(env = 3L, species = 5L, sites = 400L)

m = sLVM(com$response, com$env_weights,
         family = binomial("probit"), formula = ~0+.,posterior = "DiagonalNormal", lr = list(0.03), iter=100L, priors = list(10.0,1.0,1.0))
plot(density(m$posterior_samples$mu[,3,1]))
abline(v = quantile(m$posterior_samples$mu[,3,1], probs = c(0.025, 0.975)))
abline(v = mean((m$posterior_samples$mu[,3,1])), col="red")
abline(v = median((m$posterior_samples$mu[,3,1])), col="red")

