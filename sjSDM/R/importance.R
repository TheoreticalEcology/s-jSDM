#' importance 
#' 
#' importance of abiotic, biotic, and spatial effects
#' 
#' @param model object fitted by \code{\link{sjSDM}} or a list with beta, the association matrix, and the correlation matrix of the predictors, see details below
#' 
#' @example /inst/examples/importance-example.R
#' @author Maximilian Pichler
#' @export
importance = function(model) {
  stopifnot(
    #inherits(model, "sjSDM"),
    #inherits(model$settings$env, "linear"),
    is.null(model$settings$spatial) || inherits(model$settings$spatial, "linear")
    )
  #method = match.arg(method)
    
    if(inherits(model, "sjSDM")) {
      sp_names = colnames(model$data$Y)
    
      coefs = coef.sjSDM(model)[[1]]
      if(inherits(coefs, "list")) coefs = coefs[[1]]
      env = t(coefs)
      
      beta = env
      sigma = getCov(model)
      sigma = cov2cor(sigma)
      diag(sigma) = 0 # remove identity matrix
      
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
      
      #return(res)
    } else {
      sp_names = NULL
      
      beta = model[[1]]
      sigma = model[[2]]
      covX = model[[3]]
      
      vp = getImportance(beta = beta,  association = sigma, covX = covX)
      colnames(vp$env) = model$names
      res = list(split = vp, 
                 total = list(env = rowSums(vp$env), biotic = vp$biotic))
      #return(res)
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
#' @export
print.sjSDMimportance= function(x, ...) {
  if(is.null(x$sp_names)) print(data.frame(sp = 1:length(x$res$total$biotic), x$res$total))
  else print(data.frame(sp = x$sp_names, x$res$total))
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
#' @export
plot.sjSDMimportance= function(x, y,contour=FALSE,col.points="#24526e",cex.points=1.2,pch=19,
                           col.contour="#ffbf02", ...) {
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



#model = list(t(cbind(coef(m)$Xcoef, coef(m)$Intercept)), getResidualCov(m, FALSE)$cov, cov(m$TMBfn$env$data$x))

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
  
  PredRandom = rowSums(association^2)/sqrt(ncol(association))
  
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



#' varPartTypIII
#' 
#' variation partitioning Typ III
#' @param model object fitted by \code{\link{sjSDM}}
#' @param order which modules should be removed: E (environment), S (spatial), or B (biotic)
#' @param ... arguments passed to \code{\link{Rsquared}}
#' 
#' @author Maximilian Pichler

varPartTypIII = function(model, order = NULL, ...) {
  if(!is.null(model$spatial_weights)) {
    sp = TRUE
    order = c("ESB", "ES", "E")
  } else {
    sp = FALSE
    order = c("EB", "E")
  }
  res = vector("list", length(order)+1)
  res[[1]] = Rsquared(model, ...)
  for(i in 1:length(order))res[[i+1]] = getRsquaredWOmodule(model,modules = order[[i]],...)
  names(res) =  c("full", order)
  return(res)
}


getRsquaredWOmodule = function(model, modules = c("E"), ...) {
  modules = strsplit(modules,split = "")[[1]]
  for(i in modules) {
    if(i == "E") {
      model$model$set_env_weights(lapply(model$weights, function(w) copyRP(matrix(0.0, nrow(w), ncol(w) )) ))
    }
    if(i == "B") {
      model$model$set_sigma(copyRP(diag(1.0, ncol(model$data$Y))))
      model$model$df= as.integer(ncol(model$data$Y))
    }
    if(i == "S") {
      model$model$set_spatial_weights(lapply(model$spatial_weights, function(w) copyRP(matrix(0.0, nrow(w), ncol(w)))    ))
    }
  }
  R2 = Rsquared(model, ...)
  
  for(i in modules) {
    if(i == "E") {
      model$model$set_env_weights(copyRP(model$weights))
    }
    if(i == "B") {
      model$model$set_sigma(copyRP(model$sigma))
      model$model$df= as.integer(ncol(model$sigma))
    }
    if(i == "S") {
      model$model$set_spatial_weights(lapply(model$spatial_weights, function(w) copyRP(matrix(0.0, nrow(w), ncol(w) ))))
    }
  }
  return(R2)
}

