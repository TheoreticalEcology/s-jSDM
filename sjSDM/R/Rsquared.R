#' Rsquared2
#' 
#' calculate Rsquared following Nakagawa 
#' @param model model
#' @param X new environmental covariates
#' @param Y new species occurences
#' @param SP new spatial covariates
#' @param individual R squared for each site
#' @param ... additional parameters
#' 
#' @author Maximilian Pichler
#' @export
Rsquared2 = function(model, X = NULL, Y = NULL, SP = NULL,individual=TRUE,...) {
  
  if(model$family$link == "probit") varDist = 1
  else varDist = pi^2/3
  
  sigma = model$model$get_sigma
  df = model$settings$biotic$df
  #model$model$set_sigma(copyRP(matrix(0.0, nrow(sigma), ncol(sigma))))
  preds = apply(abind::abind(lapply(1:50, function(i) predict.sjSDM(model, link ="raw") ), along = -1L), 2:3, mean)#,newdata = X,SP=SP, link ="raw"))
  #model$model$set_sigma(copyRP(sigma))
  if(!individual) vv = stats::var(as.vector(preds))
  else vv = apply(preds,1,stats::var)
  Assocation = getCov(model)
  
  re = sum(diag(diag(1, nrow(Assocation), ncol(Assocation)) %*% Assocation))/(ncol(Assocation))
  return(list(
    marginal = vv/(vv+  re +varDist),
    conditional = (vv + re)/(vv+  re +varDist)
  ))
}




#' Rsquared
#' 
#' calculate Rsquared following Nakagawa 
#' @param model model
#' @param X new environmental covariates
#' @param Y new species occurrences
#' @param SP new spatial covariates
#' @param adjust adjust R squared or not
#' @param averageSP average R squared over species
#' @param averageSite average R squared over sites
#' 
#' @author Maximilian Pichler
#' @export

Rsquared = function(model, X = NULL, Y = NULL, SP=NULL, adjust=FALSE, averageSP = TRUE, averageSite=TRUE){
  
  if(!is.null(model$spatial_weights)) sp = TRUE
  else sp = FALSE
  
  if(is.null(X)){
    X = model$data$X
    if(sp) SP = model$settings$spatial$X
    Y = model$data$Y
  } 

  nsite = nrow(Y)
  nsp = ncol(Y)
  preds = lapply(1:100, function(i) predict.sjSDM(model, link ="raw"))#,newdata = X,SP=SP, link ="raw"))
  Ypred = apply(abind::abind(preds, along = 0L), 2:3, mean)
  link = model$family$family
  if(model$family$link == "probit") varDist = 1
  else varDist = pi^2/3
  
  #if(sp) Xvar = c(Xvar, SPvar)
  
  YMeans = matrix(colMeans(Ypred), nrow = nsite, ncol = nsp, byrow = TRUE)
  varModelSite = (Ypred - YMeans)^2/(nsite - 1)
  varModel = colSums(varModelSite)
  varAdd = diag(stats::var(Y - link$linkinv(Ypred)))
  varTot = matrix(varModel + varAdd + varDist, nrow = nsite, ncol = nsp, byrow = TRUE)#+ colSums(getCov(model)^2)
  R2 = (varModelSite)/varTot
  if(averageSite) {
    R2 = colSums(varModelSite)/varTot[1, ]
    if(averageSP) R2 = mean(R2)
  }
  else {
    if(averageSP) R2 = rowMeans(R2)
  }
  if (adjust) {
    nexp = ncol(X)
    if(sp) nexp=nexp+ncol(SP)
    if (!averageSite) {
      if(!averageSP) R2Cum = colSums(R2)
      else R2Cum = R2
      R2CumAdj = 1 - ((nsite - 3)/(nsite - nexp - 2)) * (1 - R2Cum)
      Corr = R2Cum - R2CumAdj
      R2 = R2 - Corr/nsite
    }
    else {
      R2 = 1 - ((nsite - 3)/(nsite - nexp - 2)) * (1 - R2)
    }
  }
  return(R2)
}



 