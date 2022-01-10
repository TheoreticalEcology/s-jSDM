#' Anova
#' 
#' Compute analysis of variance
#' 
#' @param object model of object \code{\link{sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' 
#' @description
#' 
#' Calculate type I anova in the following order:
#' 
#' Null, biotic, abiotic (environment), and spatial (if present).
#' 
#' Deviance for interactions (e.g. between space and environment) are also calculated and can be visualized via \code{\link{plot.sjSDManova}}.
#' 
#' @return 
#' An S3 class of type 'sjSDManova' including the following components:
#' 
#' \item{results}{Data frame of results.}
#' \item{to_print}{Data frame, summarized results for type I anova.}
#' \item{N}{Number of observations (sites).}
#' \item{spatial}{Logical, spatial model or not}
#' 
#' Implemented S3 methods are \code{\link{print.sjSDManova}} and \code{\link{plot.sjSDManova}}
#'  
#' @seealso \code{\link{plot.sjSDManova}}, \code{\link{print.sjSDManova}}
#' @import stats
#' @export

anova.sjSDM = function(object, ...) {
  out = list()
  full_m = logLik(object)[[1]]
  ### fit different models ###
  #e_form = stats::as.formula(paste0(as.character(object$settings$env$formula), collapse = ""))
  if(!inherits(object, "spatial")) {
    out$spatial = FALSE
    null_m = logLik(update(object, env_formula = ~1, biotic=bioticStruct(diag = TRUE )))[[1]]
    A_m =    logLik(update(object, env_formula = NULL, biotic=bioticStruct(diag = TRUE )))[[1]]
    B_m =    logLik(update(object, env_formula = ~1, biotic=bioticStruct(diag = FALSE)))[[1]]
    SAT_m =  logLik(update(object, env_formula = ~as.factor(1:nrow(object$data$X)), biotic=bioticStruct(diag = TRUE )))[[1]]
    
    A =    A_m
    B =    B_m
    full = full_m
    AB =   null_m - ((null_m-full_m) - (null_m - A + null_m - B))
    sat =  SAT_m
    null = null_m
    anova_rows = c("Null", "B", "Full")
    names(anova_rows) = c("Null", "Biotic", "Abiotic")
    results = data.frame(models = c("A", "B","AB", "Full", "Saturated", "Null"),
                         ll = -c(A, B,AB, full, sat, null))
  } else {
    out$spatial = TRUE
    s_form = stats::as.formula(paste0(as.character(object$settings$spatial$formula), collapse = ""))
    null_m = logLik(update(object, env_formula = ~1, spatial_formula= ~0, biotic=bioticStruct(diag = TRUE )))[[1]]
    A_m =    logLik(update(object, env_formula = NULL, spatial_formula= ~0, biotic=bioticStruct(diag = TRUE )))[[1]]
    B_m =    logLik(update(object, env_formula = ~1, spatial_formula= ~0, biotic=bioticStruct(diag = FALSE)))[[1]]
    S_m =    logLik(update(object, env_formula = ~1, spatial_formula= NULL, biotic=bioticStruct(diag = TRUE)))[[1]]
    AB_m =   logLik(update(object, env_formula = NULL, spatial_formula= ~0, biotic=bioticStruct(diag = FALSE )))[[1]]
    AS_m =   logLik(update(object, env_formula = NULL, spatial_formula= NULL, biotic=bioticStruct(diag = FALSE )))[[1]]
    BS_m =   logLik(update(object, env_formula = ~1, spatial_formula= NULL, biotic=bioticStruct(diag = TRUE )))[[1]]
    SAT_m =  logLik(update(object, env_formula = ~as.factor(1:nrow(object$data$X)), spatial_formula= ~0, biotic=bioticStruct(diag = TRUE )))[[1]]
    
    A =    A_m
    B =    B_m
    S =    S_m
    AB =   null_m - ((null_m-AB_m) - (null_m- A + null_m-B))
    AS =   null_m - ((null_m-AB_m) - (null_m- A + null_m-S))
    BS =   null_m - ((null_m-BS_m) - (null_m- B + null_m-S))
    ABS =  null_m - ((null_m-full_m) - (null_m- B + null_m-S+null_m-A))
    full = full_m
    null = null_m
    sat =  SAT_m
    
    results = data.frame(models = c("A", "B","S","B+A","B+S","A+S","AB","AS", "BS", "ABS", "Full", "Saturated", "Null"),
                         ll = -c(A, B,S,AB_m,BS_m, AS_m, AB, AS, BS, ABS, full, sat, null))
    
    
    #1-exp(2/nrow(object$data$Y)*(-A))
    anova_rows = c("Null", "B", "B+A", "Full")
    names(anova_rows) = c("Null", "Biotic", "Abiotic", "Spatial")
  }
  
  results$`Residual deviance` = -2*(results$ll - results$ll[which(results$models == "Saturated", arr.ind = TRUE)])
  
  results$Deviance = results$`Residual deviance`[which(results$models == "Null", arr.ind = TRUE)] - results$`Residual deviance`
  R2 = function(a, b) return(1-exp(2/(nrow(object$data$Y))*(-a+b)))
  results$`R2 Nagelkerke` = R2(rep(-results$ll[which(results$models == "Null", arr.ind = TRUE)], length(results$ll)), - results$ll)
  R2 = function(a, b) 1 - (b/a)
  results$`R2 McFadden`= R2(rep(-results$ll[which(results$models == "Null", arr.ind = TRUE)], length(results$ll)), - results$ll)
  
  to_print = results
  rownames(to_print) = to_print$models
  to_print = to_print[anova_rows,]
  to_print$models = names(anova_rows)
  to_print$Deviance = c(0,-diff(to_print$`Residual deviance`))
  to_print = to_print[-1,c(1, 4, 3,5,6)]
  rownames(to_print) = to_print$models
  to_print = to_print[,-1]
  out$results = results
  out$to_print = to_print
  out$N = nrow(object$data$Y)
  class(out) = "sjSDManova"
  return(invisible(out))
}




#' Print sjSDM anova
#' 
#' @param x an object of \code{\link{anova.sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' 
#' @return The above matrix is silently returned
#' 
#' 
#' @export
print.sjSDManova = function(x, ...) {
  cat("Analysis of Deviance Table\n\n")
  cat("Terms added sequentially:\n\n")
  stats::printCoefmat(x$to_print)
  return(invisible(x$to_print))
}


#' Plot anova results
#' 
#' 
#' @param x anova object from \code{\link{anova.sjSDM}}
#' @param y unused argument
#' @param type use of deviance or of Nagelkerke or McFadden R-squared
#' @param cols colors for the groups
#' @param alpha alpha for colors
#' @param ... Additional arguments to pass to \code{plot()}
#' 
#' @return The visualized matrix is silently returned
#' 
#' @export
plot.sjSDManova = function(x, y, type = c("Deviance", "Nagelkerke", "McFadden"), cols = c("#7FC97F","#BEAED4","#FDC086"),alpha=0.15, ...) {
  lineSeq = 0.3
  nseg = 100
  dr = 1.0
  type = match.arg(type)
  
  oldpar = par(no.readonly = TRUE)
  on.exit(par(oldpar))
  
  values = x$results
  select_rows = 
    if(x$spatial) { 
      sapply(c("A", "B", "AB","S", "AS", "BS", "ABS"), function(i) which(values$models == i, arr.ind = TRUE))
    } else {
      sapply(c("A", "B", "AB"), function(i) which(values$models == i, arr.ind = TRUE))
    }
  
  values = values[select_rows,]
  col_index = 
    switch (type,
      Deviance = 4,
      Nagelkerke = 5,
      McFadden = 6
    )
  
  
  graphics::plot(NULL, NULL, xlim = c(0,1), ylim =c(0,1),pty="s", axes = FALSE, xlab = "", ylab = "")
  xx = 1.1*lineSeq*cos( seq(0,2*pi, length.out=nseg))
  yy = 1.1*lineSeq*sin( seq(0,2*pi, length.out=nseg))
  graphics::polygon(xx+lineSeq,yy+(1-lineSeq), col= addA(cols[1],alpha = alpha), border = "black", lty = 1, lwd = 1)
  graphics::text(lineSeq-0.1, (1-lineSeq),labels = round(values[1,col_index],3))
  graphics::text(mean(xx+lineSeq), 0.9,labels = "Environmental", pos = 3)
  
  graphics::polygon(xx+1-lineSeq,yy+1-lineSeq, col= addA(cols[2],alpha = alpha), border = "black", lty = 1, lwd = 1)
  graphics::text(1-lineSeq+0.1, (1-lineSeq),labels = round(values[2,col_index],3))
  graphics::text(1-mean(xx+lineSeq), 0.9,labels = "Associations", pos = 3)
  graphics::text(0.5, (1-lineSeq),labels = round(values[3,col_index],3))
  
  if(x$spatial) {
    graphics::polygon(xx+0.5,yy+lineSeq, col= addA(cols[3],alpha = alpha), border = "black", lty = 1, lwd = 1)
    graphics::text(0.5, lineSeq+0.0,pos = 1,labels = round(values[4,col_index],3))
    graphics::text(0.5, 0.1,labels = "Spatial", pos = 1)
    graphics::text(0.3, 0.5,pos=1,labels   = round(values[5,col_index],3)) # AS
    graphics::text(1-0.3, 0.5,pos=1,labels = round(values[6,col_index],3)) # BS
    graphics::text(0.5, 0.5+0.05,labels    = round(values[7,col_index],3)) # ABS
  }
  return(invisible(values))
}

