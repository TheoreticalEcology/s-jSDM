#' varPart
#' 
#' variation partitioning
#' @param beta abiotic weights
#' @param sp spatial weights
#' @param covariance species associations
#' @param covX environmental covariance matrix
#' @param covSP spatial covariance matrix
#' 
#' @author Maximilian Pichler
#' @export

varPart = function(beta, sp=NULL, covariance, covX, covSP=NULL) {
  nsp = ncol(beta)
  nGroups = nrow(beta)
  
  predXTotal = rep(0, nsp)
  predXSplit = matrix(0, nrow = nsp, ncol = nGroups)
  predSPTotal = rep(0, nsp)
  if(!is.null(sp)) predSPSplit = matrix(0, nrow = nsp, ncol = nrow(sp))
  PredRandom = matrix(0, nrow = nsp, ncol = 1L)
  
  for (j in 1:nsp) {
    predXTotalSub = beta[,j] %*% crossprod(covX, beta[,j])
    predXTotal[j] <- predXTotal[j] + predXTotalSub
    for (k in 1:nGroups) {
      predXPart = beta[k, j] %*% crossprod(covX[k,k], beta[k,j])
      predXSplit[j, k] <- predXSplit[j, k] + predXPart
    }
  }
  
  if(!is.null(sp)) {
    for (j in 1:nsp) {
      predSPTotalSub = sp[,j] %*% crossprod(covSP, sp[,j])
      predSPTotal[j] <- predSPTotal[j] + predSPTotalSub
      for (k in 1:(nrow(sp))) {
        predSPPart = sp[k, j] %*% crossprod(covSP[k,k], sp[k,j])
        predSPSplit[j, k] <- predSPSplit[j, k] + predSPPart
      }
    }
  }
  
  PredRandom = rowSums(covariance^2)
  
  if(is.null(sp)) {
    variTotal = predXTotal + PredRandom
    variPartX = predXTotal/variTotal
    
    variPartRandom = PredRandom/variTotal
    variPartXSplit =  predXSplit/replicate(nGroups,apply(predXSplit, 1, sum))
    variPart = cbind(replicate(nGroups, variPartX) * variPartXSplit, variPartRandom)
    res = variPart
    return(res)
  } else {
    variTotal = predXTotal + PredRandom + predSPTotal
    variPartX = predXTotal/variTotal
    variPartSP = predSPTotal/variTotal
    
    variPartRandom = PredRandom/variTotal
    variPartXSplit =  predXSplit/replicate(nGroups,apply(predXSplit, 1, sum))
    variPartSPSplit =  predSPSplit/replicate(nrow(sp),apply(predSPSplit, 1, sum))
    variPart = cbind(replicate(nGroups, variPartX) * variPartXSplit,replicate(nrow(sp), variPartSP) * variPartSPSplit, variPartRandom)
    res = variPart
    return(res)
  }
}
