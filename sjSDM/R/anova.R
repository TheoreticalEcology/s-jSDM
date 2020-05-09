#' Anova
#' 
#' Compute analysis of variance via cross-validation for environmental, associations, and spatial effects 
#' 
#' @param object model of object \code{\link{sjSDM}}
#' @param cv number of cross-validation splits
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#'  
#' @seealso \code{\link{plot.sjSDManova}}, \code{\link{print.sjSDManova}}
#' @export

anova.sjSDM = function(object, cv = 5L,...) {
  
  splits = createSplit(nrow(object$settings$env$X), cv)
  
  out = list()
  
  if(!is.null(object$spatial_weights)){
    
    .empty = sapply(splits, function(sp) turnOn(object, modules = "", test = sp))
    empty = list(ll = sum(unlist(.empty[1,])), R = mean(unlist(.empty[2,])),R2 = mean(unlist(.empty[3,])) )
    
    .full = sapply(splits, function(sp) turnOn(object, modules = "ABS", test = sp))
    full = list(ll = sum(unlist(.full[1,])), R = mean(unlist(.full[2,])),R2 = mean(unlist(.full[3,])))
    
    .A = sapply(splits, function(sp) turnOn(object, modules = "A", test = sp))
    A = list(ll = sum(unlist(.A[1,])), R = mean(unlist(.A[2,])),R2 = mean(unlist(.A[3,])))
    
    .B = sapply(splits, function(sp) turnOn(object, modules = "B", test = sp))
    B = list(ll = sum(unlist(.B[1,])), R = mean(unlist(.B[2,])),R2 = mean(unlist(.B[3,])))
    
    .S = sapply(splits, function(sp) turnOn(object, modules = "S", test = sp))
    S = list(ll = sum(unlist(.S[1,])), R = mean(unlist(.S[2,])),R2 = mean(unlist(.S[3,])))
    
    .AB = sapply(splits, function(sp) turnOn(object, modules = "AB", test = sp))
    AB = list(ll = sum(unlist(.AB[1,])), R = mean(unlist(.AB[2,])), R2 = mean(unlist(.AB[3,])))
    
    .AS = sapply(splits, function(sp) turnOn(object, modules = "AS", test = sp))
    AS = list(ll = sum(unlist(.AS[1,])), R = mean(unlist(.AS[2,])),R2 = mean(unlist(.AS[3,])))
    
    .BS = sapply(splits, function(sp) turnOn(object, modules = "BS", test = sp))
    BS = list(ll = sum(unlist(.BS[1,])), R = mean(unlist(.BS[2,])), R2 = mean(unlist(.BS[3,])))
    
    res = data.frame(matrix(NA, 9,5))
    colnames(res) = c("Modules", "LogLik", "R2", "marginal R2", "condtional R2")    
    res$Modules = c("_", "A", "B", "S", "A+B", "A+S", "B+S", "A+B+S", "Full")
    res[1,2:4] = unlist(empty)
    
    res[2,2] = -empty$ll + A$ll # A
    res[3,2] = -empty$ll + B$ll # B
    res[4,2] = -empty$ll + S$ll # S
    
    res[5,2] = -empty$ll + AB$ll - (res[2,2] + res[3,2] )# A+B
    res[6,2] = -empty$ll + AS$ll - (res[2,2] + res[4,2] )# A+S
    res[7,2] = -empty$ll + BS$ll - (res[4,2] + res[3,2] )# B+S
    
    res[8,2] = -empty$ll + full$ll - sum(res[2:7,2]) # A+B+S
    res[9,2] = -empty$ll + full$ll # Full
    
    res[2,3] = empty$R + A$R
    res[3,3] = empty$R + B$R
    res[4,3] = empty$R + S$R
    
    res[5,3] = empty$R + AB$R - res[2,3] - res[3,3]
    res[6,3] = empty$R + AS$R - res[2,3] - res[4,3]
    res[7,3] = empty$R + BS$R - res[4,3] - res[3,3]
    res[8,3] = empty$R + full$R - sum(res[2:7,3])
    res[9,3] = empty$R + full$R
    
    
    res[2,4] = empty$R2 + A$R2
    res[3,4] = empty$R2 + B$R2
    res[4,4] = empty$R2 + S$R2
    res[5,4] = empty$R2 + AB$R2 - res[2,4] - res[3,4]
    res[6,4] = empty$R2 + AS$R2 - res[2,4] - res[4,4]
    res[7,4] = empty$R2 + BS$R2 - res[4,4] - res[3,4]
    res[8,4] = empty$R2 + full$R2 - sum(res[2:7,4])
    res[9,4] = empty$R2 + full$R2
    out$spatial = TRUE
    
  } else {
    .empty = sapply(splits, function(sp) turnOn(object, modules = "", test = sp))
    empty = list(ll = sum(unlist(.empty[1,])), R = mean(unlist(.empty[2,])), R2 = mean(unlist(.empty[3,])))
    
    .A = sapply(splits, function(sp) turnOn(object, modules = "A", test = sp))
    A = list(ll = sum(unlist(.A[1,])), R = mean(unlist(.A[2,])), R2 = mean(unlist(.A[3,])))
    
    .B = sapply(splits, function(sp) turnOn(object, modules = "B", test = sp))
    B = list(ll = sum(unlist(.B[1,])), R = mean(unlist(.B[2,])), R2 = mean(unlist(.B[3,])))
    
    .AB = sapply(splits, function(sp) turnOn(object, modules = "AB", test = sp))
    AB = list(ll = sum(unlist(.AB[1,])), R = mean(unlist(.AB[2,])), R2 = mean(unlist(.AB[3,])))
    
    
    res = data.frame(matrix(NA, 5,5))
    colnames(res) = c("Modules", "LogLik", "R2", "marginal R2", "condtional R2")    
    res$Modules = c("_", "A", "B", "A+B", "Full")
    res[1,2:4] = unlist(empty)
    
    res[2,2] = -empty$ll + A$ll # A
    res[3,2] = -empty$ll + B$ll # B
    
    res[4,2] = -empty$ll + AB$ll - (res[2,2] + res[3,2] )# A+B
    
    res[5,2] = -empty$ll + AB$ll  # Full
    
    res[2,3] = empty$R + A$R
    res[3,3] = empty$R + B$R
    
    res[4,3] = empty$R + AB$R - res[2,3] - res[3,3]
    res[5,3] = empty$R + AB$R
    
    
    res[2,4] = empty$R2 + A$R2
    res[3,4] = empty$R2 + B$R2
    res[4,4] = empty$R2 + AB$R2 - res[2,4] - res[3,4]
    res[5,4] = empty$R2 + AB$R2
    out$spatial = FALSE
  }
  out$result = res[,-5]
  class(out) = c("sjSDManova")
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


#' Plot anova
#' 
#' @param x a model fitted by \code{\link{sjSDM_cv}}
#' @param y unused argument
#' @param perf performance measurement to plot
#' @param cols colors for the groups
#' @param alpha alpha for colors
#' @param ... Additional arguments to pass to \code{plot()}
#' @export
plot.sjSDManova = function(x, y, perf = c("LogLik", "R2"),cols = c("#7FC97F","#BEAED4","#FDC086"),alpha=0.15, ...) {
  lineSeq = 0.3
  nseg = 100
  dr = 1.0
  perf = match.arg(perf)
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




turnOn = function(model, modules = c("AB"), test= NULL, ...) {
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
  env2$X = matrix(0.0, nrow(env2$X),ncol(env2$X))
  biotic2 = bioticStruct(diag = TRUE)
  spatial2 = spatial
  if(!is.null(spatial2)) {
    spatial2$X = matrix(0.0, nrow(spatial2$X),ncol(spatial2$X))
    test_sp = matrix(0.0, nrow(model$settings$spatial$X[test,,drop=FALSE]),ncol(spatial2$X))
  }
  test_env = matrix(0.0, nrow(model$settings$env$X[test,,drop=FALSE]),ncol(env2$X))
  

  for(i in modules){
    if(i == "A") {
      env2 = env
      test_env = model$settings$env$X[test,,drop=FALSE]
    }
    if(i == "B") biotic2 = model$settings$biotic
    if(i == "S") {
      spatial2 = spatial
      test_sp = model$settings$spatial$X[test,,drop=FALSE]
    }
  }
  
  m2 = sjSDM(Y = Y, 
             env = env2, 
             biotic = biotic2,
             spatial= spatial2,
             iter = model$settings$iter, 
             step_size = model$settings$iter, 
             link = model$settings$link, 
             learning_rate = model$settings$learning_rate,
             device = model$settings$device
  )
  
  if(is.null(test)) {
    return(list(ll = logLik.sjSDM(m2), R=Rsquared(m2,...)))
  } else {
    
    m2$data$X = test_env 
    m2$data$Y = model$data$Y[test,,drop=FALSE]
    
    if(!is.null(spatial)) {
      
      m2$spatial$X = test_sp
      
      return(list(ll=m2$model$logLik(X=test_env,Y=model$data$Y[test,,drop=FALSE], SP=test_sp )[[1]], 
                  R=Rsquared(model=m2,...), R2 = Rsquared2(model=m2,...)$marginal ))
    } else {

      return(list(ll=m2$model$logLik(X=test_env,Y=model$data$Y[test,,drop=FALSE])[[1]], 
                  R=Rsquared(m2,...), R2 = Rsquared2(model=m2,...)$marginal))      
      
      
    }
  }
}


