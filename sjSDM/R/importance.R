#' importance 
#' 
#' Computes standardized variance components with respect to abiotic, biotic, and spatial effect groups. 
#' 
#' @param x object fitted by \code{\link{sjSDM}} or a list with beta, the association matrix, and the correlation matrix of the predictors, see details below
#' @param save_memory use torch backend to calculate importance with single precision floats
#' @param ... additional arguments
#' 
#' @details 
#' 
#' This variance partitioning approach is based on Ovaskainen et al., 2017. For an example how to interpret the outputs, see Leibold et al., 2021.
#' This function will be deprecated in the future. Please use \code{plot(anova(model), internal=TRUE)} (currently only supported for spatial models).
#' 
#' @return
#' 
#' An S3 class of type 'sjSDMimportance' including the following components:
#' 
#' \item{names}{Character vector, species names.}
#' \item{res}{Data frame of results.}
#' \item{spatial}{Logical, spatial model or not.}
#' 
#' Implemented S3 methods include \code{\link{print.sjSDMimportance}} and \code{\link{plot.sjSDMimportance}}
#' 
#' @references 
#' 
#' Ovaskainen, O., Tikhonov, G., Norberg, A., Guillaume Blanchet, F., Duan, L., Dunson, D., ... & Abrego, N. (2017). How to make more out of community data? A conceptual framework and its implementation as models and software. Ecology letters, 20(5), 561-576.
#' 
#' Leibold, M. A., Rudolph, F. J., Blanchet, F. G., De Meester, L., Gravel, D., Hartig, F., ... & Chase, J. M. (2021). The internal structure of metacommunities. Oikos.
#' 
#' @seealso \code{\link{print.sjSDMimportance}}, \code{\link{plot.sjSDMimportance}}
#' @example /inst/examples/importance-example.R
#' @author Maximilian Pichler
#' @export
importance = function(x, save_memory = TRUE, ...) {
  model = x
  stopifnot(
    inherits(model, "sjSDM"),
    is.null(model$settings$spatial) || inherits(model$settings$spatial, "linear")
    )
  if(!save_memory) {    
      sp_names = colnames(model$data$Y)
      coefs = coef.sjSDM(model)[[1]]
      if(inherits(coefs, "list")) coefs = coefs[[1]]
      env = t(coefs)
      beta = env
      sigma = getCov(model)
      covX = stats::cov(model$data$X)
      if(!is.null(model$settings$spatial)) {
        spatial = TRUE
        sp = t(coef.sjSDM(model)[[2]][[1]])
        covSP = stats::cov(model$settings$spatial$X)
        
        vp = getImportance(beta = beta, sp = sp, association = sigma, covX = covX, covSP = covSP)
        colnames(vp$spatial) = attributes(model$settings$spatial$X)$dimnames[[2]]
        colnames(vp$env) = model$names
        res = list(split = vp, 
                   total = list(env = rowSums(vp$env), spatial = rowSums(vp$spatial), biotic = vp$biotic))
      } else {
        vp = getImportance(beta = beta,  association = sigma, covX = covX)
        colnames(vp$env) = model$names
        res = list(split = vp, 
                   total = list(env = rowSums(vp$env), biotic = vp$biotic))
        spatial=FALSE
      }
  } else {
    check_module()
    sp_names = colnames(model$data$Y)
    coefs = coef.sjSDM(model)[[1]]
    if(inherits(coefs, "list")) coefs = coefs[[1]]
    env = t(coefs)
    beta = env
    covX = stats::cov(model$data$X)
    if(!is.null(model$settings$spatial)) {
      spatial = TRUE
      sp = t(coef.sjSDM(model)[[2]][[1]])
      covSP = stats::cov(model$settings$spatial$X)
      
      vp = force_r( pkg.env$fa$importance(beta = beta, betaSP = sp, sigma = model$sigma, covX = covX, covSP = covSP, ...) )
      colnames(vp$spatial) = attributes(model$settings$spatial$X)$dimnames[[2]]
      colnames(vp$env) = model$names
      res = list(split = vp, 
                 total = list(env = rowSums(vp$env), spatial = rowSums(vp$spatial), biotic = vp$biotic))
    } else {
      vp = force_r( pkg.env$fa$importance(beta = beta,  sigma = model$sigma, covX = covX, ...) )
      colnames(vp$env) = model$names
      res = list(split = vp, 
                 total = list(env = rowSums(vp$env), biotic = vp$biotic))
      spatial=FALSE
    }
    
  }
  out = list()
  out$names = sp_names
  out$res = res
  out$spatial = spatial
  class(out) = "sjSDMimportance"
  return(out)
}

#' Print importance
#' 
#' @param x an object of \code{\link{importance}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' 
#' @return The matrix above is silently returned
#' 
#' @export
print.sjSDMimportance= function(x, ...) {
  if(is.null(x$sp_names)) res = data.frame(sp = 1:length(x$res$total$biotic), x$res$total)
  else res = data.frame(sp = x$sp_names, x$res$total)
  print(res)
  return(invisible(res))
}


#' Plot importance
#' 
#' @param x a model fitted by \code{\link{importance}}
#' @param y unused argument
#' @param contour plot contour or not
#' @param col.points point color
#' @param cex.points point size
#' @param pch point symbol
#' @param col.contour contour color
#' @param ... Additional arguments to pass to \code{plot()}
#' 
#' @return The visualized matrix is silently returned.
#' 
#' @export
plot.sjSDMimportance= function(x, y,contour=FALSE,col.points="#24526e",cex.points=1.2,pch=19,
                           col.contour="#ffbf02", ...) {
  
  oldpar = par(no.readonly = TRUE)
  on.exit(par(oldpar))
  
  if(is.null(x$sp_names)) data = data.frame(sp = 1:length(x$res$total$biotic), x$res$total)
  else data = data.frame(sp = x$sp_names, x$res$total)
  
  if(ncol(data) > 3) {
    Ternary::TernaryPlot(grid.lines = 2, 
                         axis.labels = seq(0, 1, by = 0.5), 
                         alab = 'Environmental', blab = 'Spatial', clab = 'Biotic',
                         grid.col = "grey")
    if(contour) Ternary::TernaryDensityContour(data[,2:4], resolution = 10L, col=col.contour)
    Ternary::TernaryPoints(data[,2:4], col = col.points, pch = pch, cex=cex.points)
  } else {
    graphics::barplot(t(data[,2:3]), las = 2, names.arg=data$sp)
  }
  return(invisible(data))
}


#' getImportance
#' 
#' variation partitioning with coefficients
#' @param beta abiotic weights
#' @param sp spatial weights
#' @param association species associations
#' @param covX environmental covariance matrix
#' @param covSP spatial covariance matrix
#' 
#' @author Maximilian Pichler

getImportance = function(beta, sp=NULL, association, covX, covSP=NULL) {
  nsp = ncol(beta)
  nGroups = nrow(beta)
  
  Xtotal = rep(0, nsp)
  Xsplit = matrix(0, nrow = nsp, ncol = nGroups)
  SPtotal = rep(0, nsp)
  if(!is.null(sp)) SPsplit = matrix(0, nrow = nsp, ncol = nrow(sp))
  PredRandom = matrix(0, nrow = nsp, ncol = 1L)
  
  for (j in 1:nsp) {
    predXTotalSub = beta[,j] %*% crossprod(covX, beta[,j])
    Xtotal[j] <- Xtotal[j] + predXTotalSub
    for (k in 1:nGroups) {
      predXPart = beta[k, j] %*% crossprod(covX[k,k], beta[k,j])
      Xsplit[j, k] <- Xsplit[j, k] + predXPart
    }
  }
  
  if(!is.null(sp)) {
    for (j in 1:nsp) {
      predSPTotalSub = sp[,j] %*% crossprod(covSP, sp[,j])
      SPtotal[j] <- SPtotal[j] + predSPTotalSub
      for (k in 1:(nrow(sp))) {
        predSPPart = sp[k, j] %*% crossprod(covSP[k,k], sp[k,j])
        SPsplit[j, k] <- SPsplit[j, k] + predSPPart
      }
    }
  }
  
  PredRandom = abs(rowSums(association) - diag(association)) /(ncol(association))
  
  if(is.null(sp)) {
    variTotal = Xtotal + PredRandom
    variPartX = Xtotal/variTotal
    variPartRandom = PredRandom/variTotal
    variPartXSplit =  Xsplit/replicate(nGroups,apply(Xsplit, 1, sum))
    res = list(env = replicate(nGroups, variPartX) * variPartXSplit, 
               biotic=variPartRandom)
    return(res)
  } else {
    variTotal = Xtotal + PredRandom + SPtotal
    variPartX = Xtotal/variTotal
    variPartSP = SPtotal/variTotal
    variPartRandom = PredRandom/variTotal
    variPartXSplit =  Xsplit/replicate(nGroups,apply(Xsplit, 1, sum))
    variPartSPSplit =  SPsplit/replicate(nrow(sp),apply(SPsplit, 1, sum))
    res = list(env = replicate(nGroups, variPartX) * variPartXSplit,
               spatial = replicate(nrow(sp), variPartSP) * variPartSPSplit, 
               biotic=variPartRandom)
    return(res)
  }
}