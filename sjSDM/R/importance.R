#' Importance of environmental, spatial and association components
#' 
#' Computes standardized variance components with respect to abiotic, biotic, and spatial effect groups. 
#' 
#' @param x object fitted by \code{\link{sjSDM}} or a list with beta, the association matrix, and the correlation matrix of the predictors, see details below
#' @param save_memory use torch backend to calculate importance with single precision floats
#' @param ... additional arguments
#' 
#' @details This approach is based on Ovaskainen et al., 2017, and also used in  Leibold et al., 2021. Unlike the \code{\link{anova.sjSDM}} function in the sjSDM package, importance is not calculated by explicitly switching a particular model component of and refitting the model, but essentially by setting it ineffective. 
#' 
#' Although we have no hard reasons to discourage the use of this function, we have decided in sjSDM to measure importance maninly based on a traditional ANOVA approach. We therefore recommend users to use the \code{\link{anova.sjSDM}}.
#' 
#' This function is maintained hidden for comparison / benchmarking purpose, and in case there is a need to use it in the future. If you want to access it, use sjSDM:::importance. 
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
#' @param col.points point color
#' @param cex.points point size
#' @param ... Additional arguments to pass to \code{plot()}
#' 
#' @return The visualized matrix is silently returned.
#' 
#' @export
plot.sjSDMimportance= function(x, y, col.points="#24526e",cex.points=1.2, ...) {
  
  warning("Depcreated, see ?internalStructure")
  # oldpar = par(no.readonly = TRUE)
  # on.exit(par(oldpar))
  
  # if(is.null(x$sp_names)) data = data.frame(sp = 1:length(x$res$total$biotic), x$res$total)
  # else data = data.frame(sp = x$sp_names, x$res$total)
  # 
  # if(ncol(data) > 3) {
  #   
  #   x = as.data.frame(do.call(cbind, x$res$total))
  #   
  #   plt = 
  #     ggtern::ggtern(x, ggplot2::aes_string(x = "env", z = "spatial", y = "biotic"))+
  #     ggtern::scale_T_continuous(limits=c(0,1),
  #                                breaks=seq(0, 1,by=0.2),
  #                                labels=seq(0,1, by= 0.2)) +
  #     ggtern::scale_L_continuous(limits=c(0,1),
  #                                breaks=seq(0, 1,by=0.2),
  #                                labels=seq(0, 1,by=0.2)) +
  #     ggtern::scale_R_continuous(limits=c(0,1),
  #                                breaks=seq(0, 1,by=0.2),
  #                                labels=seq(0, 1,by=0.2)) +
  #     ggplot2::geom_point(color = col.points, size = cex.points) + 
  #     ggplot2::labs(x = "E",
  #                   xarrow = "Environment",
  #                   y = "C",
  #                   yarrow = "Species associations",
  #                   z = "S", 
  #                   zarrow = "Space") +
  #     ggtern::theme_bw() +
  #     ggtern::theme_showarrows() +
  #     ggtern::theme_arrowlong() +
  #     ggplot2::theme(
  #       panel.grid = ggplot2::element_line(color = "darkgrey", size = 0.3),
  #       plot.tag = ggplot2::element_text(size = 11),
  #       plot.title = ggplot2::element_text(size = 11, hjust = 0.1 , margin = ggplot2::margin(t = 10, b = -20)),
  #       tern.axis.arrow = ggplot2::element_line(size = 1),
  #       tern.axis.arrow.text = ggplot2::element_text(size = 6),
  #       axis.text = ggplot2::element_text(size = 4),
  #       axis.title = ggplot2::element_text(size = 6),
  #       legend.text = ggplot2::element_text(size = 6),
  #       legend.title = ggplot2::element_text(size = 8),
  #       strip.text = ggplot2::element_text(size = 8),
  #       #plot.margin = unit(c(top,1,1,1)*0.2, "cm"),
  #       strip.background = ggplot2::element_rect(color = NA),
  #     ) 
  #   plt
  #   
  # } else {
  #   graphics::barplot(t(data[,2:3]), las = 2, names.arg=data$sp)
  # }
  # return(invisible(data))
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