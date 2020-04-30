#' sLVM
#' scalable LVM model
#' 
#' @param Y species occurrences
#' @param X environmental (abiotic) covariates
#' @param formula formula for environment
#' @param lv number of latent variables
#' @param priors list of scale priors for beta, lv, and lf
#' @param posterior type of posterior distribution
#' @param iter number of optimization steps
#' @param step_size batch_size
#' @param lr learning_rate, can be also a list (for each parameter type)
#' 
#' @export
sLVM = function(Y = NULL, X = NULL, formula = NULL, lv = 2L, 
                priors = list(3.0, 1.0, 1.0), posterior = c("Delta", "LaplaceApproximation", "LowRankMultivariateNormal", "DiagonalNormal"),
                iter = 100L, step_size=20L, lr=list(0.1), device = "cpu", dtype = "float32") {
  
  check_module()
  
  out = list()
  
  if(is.numeric(device)) device = as.integer(device)
  
  if(device == "gpu") device = 0L
  
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
  
  posterior = match.arg(match.arg)
  
  lv = as.integer(lv)
  
  out$get_model = function(){
    model = fa$Model_LVM(device=device, dtype=dtype)
  }
  model = out$get_model()
  time = system.time({model$fit(X, Y, guide=posterior, scale_mu=prior[1], scale_lf=prior[2],scale_lv=prior[3],lr =lr, batch_size = step_size, epochs = as.integer(iter), parallel = 0L)})[3]
  out$model = model
  out$posterior=posterior
  out$prior=prior
  
  
}