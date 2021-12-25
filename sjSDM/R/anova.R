#' Anova
#' 
#' Compute analysis of variance via cross-validation for environmental, associations, and spatial effects 
#' 
#' @param object model of object \code{\link{sjSDM}}
#' @param cv number of cross-validation splits
#' @param individual compute analysis of variance on species and site level
#' @param sampling number of sampling steps for Monte Carlo integration
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#'  
#' @seealso \code{\link{plot.sjSDManova}}, \code{\link{print.sjSDManova}}
#' @export

anova.sjSDM = function(object, cv = 5L,individual=FALSE, sampling = 5000L, ...) {
  
  if(is.logical(cv)) cv = 0
  
  if(cv > 0) splits = createSplit(nrow(object$settings$env$X), cv)
  else splits = list(NULL)
  
  out = list()
  
  
  if(!individual) {
  
    fit_and_form = function(mod) {
      .tmp = sapply(splits, function(sp) turnOn(object, modules = mod, test = sp, individual = individual, sampling = sampling))
      return(list(ll = sum(unlist(.tmp[1,])), R = mean(unlist(.tmp[2,])),R2 = mean(unlist(.tmp[3,])) ))
    }
  
  } else {
    
    fit_and_form = function(mod) {
      .tmp = lapply(splits, function(sp) turnOn(object, modules = mod, test = sp, individual = individual, sampling = sampling))
      return(list(ll = unlist(lapply(.tmp, function(l) l$ll)), 
                  R =  unlist(lapply(.tmp, function(l) l$R)),
                  R2 = unlist(lapply(.tmp, function(l) l$R2)) ))
    }
    
  }
  
  if(!is.null(object$spatial_weights)){
    modules = list("", "ABS", "A", "B", "S", "AB", "AS", "BS")
    results = lapply(modules, function(m) fit_and_form(m))
    names(results) = c("empty", "full","A", "B", "S", "AB", "AS", "BS")
    
    if(!individual) {
    
      res = data.frame(matrix(NA, 9,5))
      colnames(res) = c("Modules", "LogLik","R2", "marginal R2", "condtional R2")    
      res$Modules = c("_", "A", "B", "S", "A+B", "A+S", "B+S", "A+B+S", "Full")
      res[1,2:4] = unlist(results$empty)
      
      res[2,2] = -results$empty$ll + results$A$ll # A
      res[3,2] = -results$empty$ll + results$B$ll # B
      res[4,2] = -results$empty$ll + results$S$ll # S
      res[5,2] = -results$empty$ll + results$AB$ll - (res[2,2] + res[3,2] )# A+B
      res[6,2] = -results$empty$ll + results$AS$ll - (res[2,2] + res[4,2] )# A+S
      res[7,2] = -results$empty$ll + results$BS$ll - (res[4,2] + res[3,2] )# B+S
      res[8,2] = -results$empty$ll + results$full$ll - sum(res[2:7,2]) # A+B+S
      res[9,2] = -results$empty$ll + results$full$ll # Full
      res[2,3] = results$empty$R + results$A$R
      res[3,3] = results$empty$R + results$B$R
      res[4,3] = results$empty$R + results$S$R
      res[5,3] = results$empty$R + results$AB$R - res[2,3] - res[3,3]
      res[6,3] = results$empty$R + results$AS$R - res[2,3] - res[4,3]
      res[7,3] = results$empty$R + results$BS$R - res[4,3] - res[3,3]
      res[8,3] = results$empty$R + results$full$R - sum(res[2:7,3])
      res[9,3] = results$empty$R + results$full$R
      res[2,4] = results$empty$R2 + results$A$R2
      res[3,4] = results$empty$R2 + results$B$R2
      res[4,4] = results$empty$R2 + results$S$R2
      res[5,4] = results$empty$R2 + results$AB$R2 - res[2,4] - res[3,4]
      res[6,4] = results$empty$R2 + results$AS$R2 - res[2,4] - res[4,4]
      res[7,4] = results$empty$R2 + results$BS$R2 - res[4,4] - res[3,4]
      res[8,4] = results$empty$R2 + results$full$R2 - sum(res[2:7,4])
      res[9,4] = results$empty$R2 + results$full$R2
      res$R2ll = res$LogLik
      res$R2ll[-1] = res$LogLik[-1] + res$LogLik[1]
      res$R2ll = 1 - (res$R2ll / res$R2ll[1])
      out$spatial = TRUE
    } else {
      
      res = array(NA, dim = c(nrow(object$data$Y),9,5))
      #colnames(res) = c("Modules", "LogLik", "R2", "marginal R2", "condtional R2")    
      #res$Modules = c("_", "A", "B", "S", "A+B", "A+S", "B+S", "A+B+S", "Full")
      res[,1,1:3] = do.call(cbind, results$empty)
      
      res[,2,1] = -results$empty$ll + results$A$ll # A
      res[,3,1] = -results$empty$ll + results$B$ll # B
      res[,4,1] = -results$empty$ll + results$S$ll # S
      res[,5,1] = -results$empty$ll + results$AB$ll - (res[,2,1] + res[,3,1] )# A+B
      res[,6,1] = -results$empty$ll + results$AS$ll - (res[,2,1] + res[,4,1] )# A+S
      res[,7,1] = -results$empty$ll + results$BS$ll - (res[,4,1] + res[,3,1] )# B+S
      res[,8,1] = -results$empty$ll + results$full$ll - sum(res[,2:7,1]) # A+B+S
      res[,9,1] = -results$empty$ll + results$full$ll # Full
      res[,2,2] = results$empty$R + results$A$R
      res[,3,2] = results$empty$R + results$B$R
      res[,4,2] = results$empty$R + results$S$R
      res[,5,2] = results$empty$R + results$AB$R - res[,2,2] - res[,3,2]
      res[,6,2] = results$empty$R + results$AS$R - res[,2,2] - res[,4,2]
      res[,7,2] = results$empty$R + results$BS$R - res[,4,2] - res[,3,2]
      res[,8,2] = results$empty$R + results$full$R - sum(res[,2:7,2])
      res[,9,2] = results$empty$R + results$full$R
      res[,2,3] = results$empty$R2 + results$A$R2
      res[,3,3] = results$empty$R2 + results$B$R2
      res[,4,3] = results$empty$R2 + results$S$R2
      res[,5,3] = results$empty$R2 + results$AB$R2 - res[,2,3] - res[,3,3]
      res[,6,3] = results$empty$R2 + results$AS$R2 - res[,2,3] - res[,4,3]
      res[,7,3] = results$empty$R2 + results$BS$R2 - res[,4,3] - res[,3,3]
      res[,8,3] = results$empty$R2 + results$full$R2 - sum(res[,2:7,3])
      res[,9,3] = results$empty$R2 + results$full$R2
      res[,,5] = res[,,1]
      res[,-1,5] = res[,-1,5] + res[,1,5]
      res[,,5] = 1 - (res[,,5] / res[,1,5])
      
    }
    
  } else {
    modules = list("","A", "B", "AB")
    results = lapply(modules, function(m) fit_and_form(m))
    names(results) = c("empty", "A", "B", "AB")
    
    if(!individual) {
      res = data.frame(matrix(NA, 5,5))
      colnames(res) = c("Modules", "LogLik", "R2", "marginal R2", "condtional R2")    
      res$Modules = c("_", "A", "B", "A+B", "Full")
      res[1,2:4] = unlist(results$empty)
      res[2,2] = -results$empty$ll + results$A$ll # A
      res[3,2] = -results$empty$ll + results$B$ll # B
      res[4,2] = -results$empty$ll + results$AB$ll - (res[2,2] + res[3,2] )# A+B
      res[5,2] = -results$empty$ll + results$AB$ll  # Full
      res[2,3] = results$empty$R + results$A$R
      res[3,3] = results$empty$R + results$B$R
      res[4,3] = results$empty$R + results$AB$R - res[2,3] - res[3,3]
      res[5,3] = results$empty$R + results$AB$R
      res[2,4] = results$empty$R2 +results$A$R2
      res[3,4] = results$empty$R2 +results$B$R2
      res[4,4] = results$empty$R2 +results$AB$R2 - res[2,4] - res[3,4]
      res[5,4] = results$empty$R2 +results$AB$R2
      res$R2ll = res$LogLik
      res$R2ll[-1] = res$LogLik[-1] + res$LogLik[1]
      res$R2ll = 1 - (res$R2ll / res$R2ll[1])
      out$spatial = FALSE
    } else {
      
      res = array(NA, dim = c(nrow(object$data$Y),5,5))
      #colnames(res) = c("Modules", "LogLik", "R2", "marginal R2", "condtional R2")    
      #res$Modules = c("_", "A", "B", "S", "A+B", "A+S", "B+S", "A+B+S", "Full")
      res[,1,1:3] = do.call(cbind, results$empty)
      
      res[,2,2-1] = -results$empty$ll + results$A$ll # A
      res[,3,2-1] = -results$empty$ll + results$B$ll # B
      res[,4,2-1] = -results$empty$ll + results$AB$ll - (res[,2,1] + res[,3,1] )# A+B
      res[,5,2-1] = -results$empty$ll + results$AB$ll  # Full
      res[,2,3-1] = results$empty$R + results$A$R
      res[,3,3-1] = results$empty$R + results$B$R
      res[,4,3-1] = results$empty$R + results$AB$R - res[,2,2] - res[,3,2]
      res[,5,3-1] = results$empty$R + results$AB$R
      res[,2,4-1] = results$empty$R2 +results$A$R2
      res[,3,4-1] = results$empty$R2 +results$B$R2
      res[,4,4-1] = results$empty$R2 +results$AB$R2 - res[,2,3] - res[,3,3]
      res[,5,4-1] = results$empty$R2 +results$AB$R2
      
      res[,,5] = res[,,1]
      res[,-1,5] = res[,-1,5] + res[,1,5]
      res[,,5] = 1 - (res[,,5] / res[,1,5])
      out$spatial = FALSE
    }
  }
  if(!individual) out$result = res[,-5]
  else out$result = res
  
  if(!individual) class(out) = c("sjSDManova")
  else class(out) = c("sjSDManovaIndividual")
  return(out)
}

#' Print sjSDM anova
#' 
#' @param x an object of \code{\link{anova.sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
print.sjSDManova = function(x, ...) {
  cat("Changes relative to empty model (without modules):\n\n")
  print(x$result,row.names = FALSE)
}


#' Print sjSDM anova
#' 
#' @param x an object of \code{\link{anova.sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
print.sjSDManovaIndividual = function(x, ...) {
  cat("Site specific changes relative to empty model (without modules), see plot(x)")
}


#' Plot anova
#' 
#' @param x anova object from \code{\link{anova.sjSDM}}
#' @param y unused argument
#' @param perf performance measurement to plot
#' @param cols colors for the groups
#' @param alpha alpha for colors
#' @param percent use relative instead of absolute values (currently not supported)
#' @param ... Additional arguments to pass to \code{plot()}
#' @export
plot.sjSDManova = function(x, y, perf = c("LogLik", "R2"),cols = c("#7FC97F","#BEAED4","#FDC086"),alpha=0.15,percent=TRUE, ...) {
  lineSeq = 0.3
  nseg = 100
  dr = 1.0
  perf = match.arg(perf)
  
  #if(percent) x$result[,-1] = x$result[,-1] / do.call(rbind, rep(list(x$result[nrow(x$result),-1]), nrow(x$result)))
  #if(percent)
  if(perf == "LogLik") perf = "R2ll"
  graphics::plot(NULL, NULL, xlim = c(0,1), ylim =c(0,1),pty="s", axes = FALSE, xlab = "", ylab = "")
  xx = 1.1*lineSeq*cos( seq(0,2*pi, length.out=nseg))
  yy = 1.1*lineSeq*sin( seq(0,2*pi, length.out=nseg))
  graphics::polygon(xx+lineSeq,yy+(1-lineSeq), col= addA(cols[1],alpha = alpha), border = "black", lty = 1, lwd = 1)
  graphics::text(lineSeq-0.1, (1-lineSeq),labels = round(x$result[[perf]][x$result$Modules=="A"], 3))
  graphics::text(mean(xx+lineSeq), 0.9,labels = "Environmental", pos = 3)
  
  graphics::polygon(xx+1-lineSeq,yy+1-lineSeq, col= addA(cols[2],alpha = alpha), border = "black", lty = 1, lwd = 1)
  graphics::text(1-lineSeq+0.1, (1-lineSeq),labels = round(x$result[[perf]][x$result$Modules=="B"], 3))
  graphics::text(1-mean(xx+lineSeq), 0.9,labels = "Associations", pos = 3)
  graphics::text(0.5, (1-lineSeq),labels = round(x$result[[perf]][x$result$Modules=="A+B"], 3))
  
  if(x$spatial) {
    graphics::polygon(xx+0.5,yy+lineSeq, col= addA(cols[3],alpha = alpha), border = "black", lty = 1, lwd = 1)
    graphics::text(0.5, lineSeq+0.0,pos = 1,labels = round(x$result[[perf]][x$result$Modules=="S"], 3))
    graphics::text(0.5, 0.1,labels = "Spatial", pos = 1)
    graphics::text(0.5, (1-lineSeq),labels = round(x$result[[perf]][x$result$Modules=="A+B"], 3))
    graphics::text(0.3, 0.5,pos=1,labels = round(x$result[[perf]][x$result$Modules=="A+S"], 3))
    graphics::text(1-0.3, 0.5,pos=1,labels = round(x$result[[perf]][x$result$Modules=="B+S"], 3))
    graphics::text(0.5, 0.5+0.05,labels = round(x$result[[perf]][x$result$Modules=="A+B+S"], 3))
  }
}

#' Plot anova
#' 
#' @param x anova object from \code{\link{anova.sjSDM}}
#' @param y unused argument
#' @param contour plot contour or not
#' @param col.points point color
#' @param cex.points point size
#' @param pch point symbol
#' @param ... Additional arguments to pass to \code{plot()}
#' @export
plot.sjSDManovaIndividual= function(x, y,contour=FALSE,col.points="#24526e",cex.points=1.2,pch=19, ...) {
  
  print("not yet supported")
  # if(dim(an$result)[2] > 5 ) {
  #   
  #   #tt = x$result[,9,]
  #   #attributes(tt) = list(dim = c(dim(x$result)[1], 1, 4))
  #   
  #   
  #   
  #   #x$result = x$result / abind::abind(rep(list(tt), 9), along = 2L)
  #   data = x$result[,2:4,1] /apply(x$result[,9,],2,sum)[1]
  #   Ternary::TernaryPlot(grid.lines = 2, 
  #                        axis.labels = c(round(min(data), 3), "", round(max(data), 3)), 
  #                        alab = 'Environmental', blab = 'Spatial', clab = 'Biotic',
  #                        grid.col = "grey")
  #   data = scales::rescale(data)
  #   
  #   Ternary::TernaryPoints(data,  col = col.points, pch = pch, cex=cex.points)
  # } else {
  #   graphics::barplot(t(data[,2:3]), las = 2, names.arg=data$sp)
  # }
  # return(invisible(data))
  
}


turnOn = function(model, modules = c("AB"), test= NULL,individual=FALSE, sampling = 1000L, ...) {
  modules = strsplit(modules,split = "")[[1]]
  
  env = model$settings$env
  spatial = model$settings$spatial
  
  if(!is.null(test)) {
    env$X = env$X[-test,,drop=FALSE]
    if(!is.null(spatial)) {
      spatial$X = spatial$X[-test,,drop=FALSE]
    }
    Y = model$data$Y[-test,,drop=FALSE]
  } else {
    Y = model$data$Y
  }
  env2 = env
  
  
  # if(length(modules) == 0) env2$X = cbind(env2$X[,1,drop=FALSE], matrix(0.0, nrow(env2$X),ncol(env2$X)-1L ))
  # else 
  env2$X =  matrix(0.0, nrow(env2$X),ncol(env2$X))
  
  biotic2 = bioticStruct(diag = TRUE)
  spatial2 = spatial
  if(!is.null(spatial2)) {
    spatial2$X = matrix(0.0, nrow(spatial2$X),ncol(spatial2$X))
    if(!is.null(test)) test_sp = matrix(0.0, nrow(model$settings$spatial$X[test,,drop=FALSE]),ncol(spatial2$X))
  }
  if(!is.null(test)) {
    # if(length(modules) == 0) test_env = cbind(matrix(1.0,  nrow(model$settings$env$X[test,,drop=FALSE]),1), 
    #                                           matrix(0.0, nrow(model$settings$env$X[test,,drop=FALSE]),ncol(env2$X)-1))
    test_env = matrix(0.0, nrow(model$settings$env$X[test,,drop=FALSE]),ncol(env2$X))
    #test_env = matrix(0.0, nrow(model$settings$env$X[test,,drop=FALSE]),ncol(env2$X))
    
  }
  

  for(i in modules){
    if(i == "A") {
      env2 = env
      if(!is.null(test)) test_env = model$settings$env$X[test,,drop=FALSE]
    }
    if(i == "B") biotic2 = model$settings$biotic
    if(i == "S") {
      spatial2 = spatial
      if(!is.null(test)) test_sp = model$settings$spatial$X[test,,drop=FALSE]
    }
  }
  
  m2 = sjSDM(Y = Y, 
             env = env2, 
             biotic = biotic2,
             spatial= spatial2,
             iter = model$settings$iter, 
             step_size = model$settings$step_size, 
             family = model$family$family, 
             learning_rate = model$settings$learning_rate,
             device = model$settings$device,
             sampling= sampling
  )
  
  if(!individual) mean_func = function(f) mean(sapply(1:50, function(i) f() ))
  else mean_func = function(f) apply(do.call(cbind, lapply(1:50, function(i) f() ) ), 1, mean)
    
  if(is.null(test)) {
    if(is.null(spatial)) {
      return(list(ll = mean_func( function() force_r( m2$model$logLik(X=m2$data$X,Y=m2$data$Y,individual=individual, sampling= sampling) )[[1]] ),  
                  R=Rsquared(model=m2,averageSite=!individual,...),
                  R2 = Rsquared2(model=m2,individual=individual,...)$marginal ))
    } else {
      return(list(ll = mean_func( function() force_r( m2$model$logLik(X=m2$data$X,Y=m2$data$Y, SP=m2$settings$spatial$X,individual=individual, sampling = sampling ) )[[1]] ),  
                  R=Rsquared(model=m2,averageSite=!individual,...),
                  R2 = Rsquared2(model=m2,individual=individual,...)$marginal ))
    }
  } else {
    
    m2$data$X = test_env 
    m2$data$Y = model$data$Y[test,,drop=FALSE]
    
    if(!is.null(spatial)) {
      
      m2$spatial$X = test_sp
      
      return(list(ll= mean_func( function() force_r( m2$model$logLik(X=test_env,Y=model$data$Y[test,,drop=FALSE], SP=test_sp,individual=individual, sampling = sampling ) )[[1]] ), 
                  R=Rsquared(model=m2,averageSite=!individual,...), 
                  R2 = Rsquared2(model=m2,individual=individual,...)$marginal ))
    } else {

      return(list(ll= mean_func( function() force_r( m2$model$logLik(X=test_env,Y=model$data$Y[test,,drop=FALSE],individual=individual, sampling = sampling) )[[1]] ), 
                  R=Rsquared(m2,averageSite=!individual,...), 
                  R2 = Rsquared2(model=m2,individual=individual,...)$marginal))      
      
      
    }
  }
}

