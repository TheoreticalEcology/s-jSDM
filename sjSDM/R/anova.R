#' Anova
#' 
#' Compute analysis of variance
#' 
#' @param object model of object \code{\link{sjSDM}}
#' @param samples Number of Monte Carlo samples
#' @param ... optional arguments which are passed to the calculation of the logLikelihood
#' 
#' @description
#' 
#' Calculate type II anova.
#' 
#' 
#' Shared contributions (e.g. between space and environment) are also calculated and can be visualized via \code{\link{plot.sjSDManova}}.
#' 
#' @return 
#' An S3 class of type 'sjSDManova' including the following components:
#' 
#' \item{results}{Data frame of results.}
#' \item{to_print}{Data frame, summarized results for type I anova.}
#' \item{N}{Number of observations (sites).}
#' \item{spatial}{Logical, spatial model or not.}
#' \item{species}{individual species R2s.}
#' \item{sites}{individual site R2s.}
#' \item{lls}{individual site by species negative-log-likelihood values.}
#' 
#' Implemented S3 methods are \code{\link{print.sjSDManova}} and \code{\link{plot.sjSDManova}}
#'  
#' @seealso \code{\link{plot.sjSDManova}}, \code{\link{print.sjSDManova}}, \code{\link{plotInternalStructure}}
#' @import stats
#' @export

anova.sjSDM = function(object, samples = 5000L, ...) {
  out = list()
  individual = TRUE
  samples = as.integer(samples)
  object = checkModel(object)
  
  if(object$family$family$family == "gaussian") stop("gaussian not yet supported")
  
  object$settings$se = FALSE
  
  null_m = -get_null_ll(object)
  
  full_m = get_conditional_lls(object, null_m, sampling = samples, ...)
  
  ### fit different models ###
  #e_form = stats::as.formula(paste0(as.character(object$settings$env$formula), collapse = ""))
  if(!inherits(object, "spatial")) {
    out$spatial = FALSE
    
    m = update(object, env_formula = NULL, spatial_formula= ~0, biotic=bioticStruct(diag = TRUE ))
    A_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = ~0, spatial_formula= ~0, biotic=bioticStruct(diag = FALSE))
    B_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = ~as.factor(1:nrow(object$data$X)), spatial_formula= ~0, biotic=bioticStruct(diag = FALSE ))
    SAT_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    
    F_A  = (null_m - full_m) - B_m
    F_B  = (null_m - full_m) - A_m
    F_AB =  (null_m - full_m) - F_A - F_B
    
    full = full_m
    null = null_m
    sat =  SAT_m
    anova_rows = c("Null", "F_A", "F_B", "Full")
    names(anova_rows) = c("Null", "Abiotic", "Biotic", "Full")
    results = data.frame(models = c("F_A", "F_B","F_AB","A","B", "Full", "Saturated", "Null"),
                         ll = -c(sum(null_m) - sum(F_A), sum(null_m) -sum(F_B), sum(null_m) - sum(F_AB),sum(A_m),sum(B_m), sum(full), sum(sat), sum(null)))
    results_ind = list("F_A"=-(null_m - F_A), "F_B"=-(null_m -F_B), "F_AB"=-(null_m -F_AB),
                       "A"=-A_m, "B"=-B_m,
                       "Full"=-full, "Saturated"=-sat, "Null"=-null)
    
  } else {
    out$spatial = TRUE
    zero_like = function(M) {
      return(matrix(0.0, nrow(M), ncol(M)))
    }
    
    s_form = stats::as.formula(paste0(as.character(object$settings$spatial$formula), collapse = ""))
    m = update(object, env_formula = NULL, spatial_formula= ~0, biotic=bioticStruct(diag = TRUE ))
    A_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = ~0, spatial_formula= ~0, biotic=bioticStruct(diag = FALSE))
    B_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = ~1, spatial_formula= NULL, biotic=bioticStruct(diag = TRUE))
    S_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = NULL, spatial_formula= ~0, biotic=bioticStruct(diag = FALSE ))
    AB_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = NULL, spatial_formula= NULL, biotic=bioticStruct(diag = TRUE ))
    AS_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = ~1, spatial_formula= NULL, biotic=bioticStruct(diag = FALSE ))
    BS_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = ~as.factor(1:nrow(object$data$X)), spatial_formula= ~0, biotic=bioticStruct(diag = FALSE ))
    SAT_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    
    F_AB =   S_m - full_m
    F_AS =   B_m - full_m
    F_BS =   A_m - full_m
    F_A  = (null_m - full_m) - F_BS
    F_B  = (null_m - full_m) - F_AS
    F_S  = (null_m - full_m) - F_AB
    F_ABS =  (null_m - full_m) - F_BS - F_AB - F_AS - F_A - F_B - F_S
    
    full = full_m
    null = null_m
    sat =  SAT_m
    
    results = data.frame(models = c("F_A", "F_B","F_S","F_AB","F_AS", "F_BS", "F_ABS",
                                    "A", "B", "S", "AB", "AS", "BS", "Full", "Saturated", "Null"),
                         ll = -c(sum(null_m) - sum(F_A), sum(null_m) - sum(F_B),sum(null_m) - sum(F_S), 
                                 sum(null_m) - sum(F_AB), sum(null_m) - sum(F_AS), sum(null_m) - sum(F_BS), sum(null_m) - sum(F_ABS), 
                                 sum(A_m), sum(B_m), sum(S_m), sum(AB_m), sum(AS_m), sum(BS_m),
                                 sum(full), sum(sat), sum(null)))
    
    results_ind = list("F_A"=-(null_m - F_A), "F_B"=-(null_m -F_B),"F_S"=-(null_m -F_S), "F_AB"=-(null_m -F_AB),"F_AS"=-(null_m -F_AS), "F_BS"=-(null_m -F_BS), "F_ABS"=-(null_m -F_ABS), 
                       "A"=-A_m, "B"=-B_m,"S"=-S_m, "AB"=-AB_m,"AS"=-AS_m, "BS"=-BS_m,
                       "Full"=-full, "Saturated"=-sat, "Null"=-null)
    
    anova_rows = c("Null", "F_A", "F_B", "F_S", "Full")
    names(anova_rows) = c("Null", "Abiotic", "Biotic", "Spatial", "Full")
  }
  
  results$`Residual deviance` = -2*(results$ll - results$ll[which(results$models == "Saturated", arr.ind = TRUE)])
  
  results$Deviance = results$`Residual deviance`[which(results$models == "Null", arr.ind = TRUE)] - results$`Residual deviance`
  R21 = function(a, b) return(1-exp(2/(nrow(object$data$Y))*(-a+b)))
  results$`R2 Nagelkerke` = R21(rep(-results$ll[which(results$models == "Null", arr.ind = TRUE)], length(results$ll)), - results$ll)
  R22 = function(a, b) 1 - (b/a)
  results$`R2 McFadden`= R22(rep(-results$ll[which(results$models == "Null", arr.ind = TRUE)], length(results$ll)), - results$ll)
  
  # individual
  Residual_deviance_ind = lapply(results_ind, function(r) r - results_ind$Saturated)
  Deviance_ind = lapply(Residual_deviance_ind, function(r) r - Residual_deviance_ind$Null)
  R211 = function(a, b, n=1) return(1-exp(2/(n)*(-a+b)))   # divide by what?
  R2_Nagelkerke_ind = lapply(results_ind, function(r) R211(-colSums(results_ind$Null), -colSums(r), n=nrow(object$data$Y)))
  R2_Nagelkerke_sites = lapply(results_ind, function(r) R211(-rowSums(results_ind$Null), -rowSums(r), n=ncol(object$data$Y)))
  R222 = function(a, b) 1 - (b/a)
  R2_McFadden_ind = lapply(results_ind, function(r) R222(-colSums(results_ind$Null), -colSums(r)))
  R2_McFadden_sites = lapply(results_ind, function(r) R222(-rowSums(results_ind$Null), -rowSums(r)))
  
  R2_McFadden_ind_shared = get_shared_anova(R2_McFadden_ind)
  R2_McFadden_sites_shared = get_shared_anova(R2_McFadden_sites)
  R2_Nagelkerke_ind_shared = get_shared_anova(R2_Nagelkerke_ind)
  R2_Nagelkerke_sites_shared = get_shared_anova(R2_Nagelkerke_sites)
  
  to_print = results
  rownames(to_print) = to_print$models
  to_print = to_print[anova_rows,]
  to_print$models = names(anova_rows)
  to_print = to_print[-1,c(1, 4, 3,5,6)]
  rownames(to_print) = to_print$models
  to_print = to_print[,-1]
  out$results = results
  out$to_print = to_print
  out$N = nrow(object$data$Y)
  out$species = list(Residual_deviance = Residual_deviance_ind,
                     Deviance = Deviance_ind,
                     R2_Nagelkerke = R2_Nagelkerke_ind,
                     R2_McFadden = R2_McFadden_ind,
                     R2_Nagelkerke_shared = R2_Nagelkerke_ind_shared,
                     R2_McFadden_shared = R2_McFadden_ind_shared                
                     )
  out$sites = list(R2_Nagelkerke = R2_Nagelkerke_sites,
                   R2_McFadden = R2_McFadden_sites,
                   R2_Nagelkerke_shared = R2_Nagelkerke_sites_shared,
                   R2_McFadden_shared = R2_McFadden_sites_shared)
  out$lls = list(results_ind)
  class(out) = "sjSDManova"
  return(invisible(out))
}

get_conditional_lls = function(m, null_m, ...) {
  joint_ll = rowSums( logLik(m, individual = TRUE, ...)[[1]] )
  raw_ll = 
    sapply(1:ncol(m$data$Y), function(i) {
      reticulate::py_to_r(
        pkg.env$fa$MVP_logLik(m$data$Y[,-i], 
                              predict(m, type = "raw")[,-i], 
                              reticulate::py_to_r(m$model$get_sigma)[-i,],
                              device = m$model$device,
                              individual = TRUE,
                              dtype = m$model$dtype,
                              batch_size = as.integer(m$settings$step_size),
                              alpha = m$model$alpha,
                              link = m$family$link,
                              theta = m$theta[-i],
                              ...
                              )
        )
    })
  raw_conditional_ll = (joint_ll - raw_ll )
  diff_ll = colSums(null_m - raw_conditional_ll)
  rates = diff_ll/sum(diff_ll)
  rescaled_conditional_lls = null_m - matrix(rates, nrow = nrow(m$data$Y), ncol = ncol(m$data$Y), byrow = TRUE) * (rowSums(null_m)-joint_ll)
  return(rescaled_conditional_lls)
}

get_shared_anova = function(R2objt, spatial = TRUE) {
  if(spatial) {
    F_BS = R2objt$Full-R2objt$A
    F_AB = R2objt$Full-R2objt$S
    F_AS = R2objt$Full-R2objt$B
    F_A = R2objt$Full-R2objt$BS
    F_B =  R2objt$Full-R2objt$AS
    F_S =  R2objt$Full-R2objt$AB
    F_BS = F_BS - F_B -F_S
    F_AB = F_AB - F_A -F_B
    F_AS = F_AS - F_A -F_S
    F_ABS = R2objt$Full - F_BS - F_AB- F_AS- F_A- F_B - F_S
    A = F_A + F_AB*abs(F_A)/(abs(F_A)+abs(F_B)) + F_AS*abs(F_A)/(abs(F_S)+abs(F_A))+ F_ABS*abs(F_A)/(abs(F_A)+abs(F_B)+abs(F_S))
    B = F_B + F_AB*abs(F_B)/(abs(F_A)+abs(F_B)) + F_BS*abs(F_B)/(abs(F_S)+abs(F_B))+ F_ABS*abs(F_B)/(abs(F_A)+abs(F_B)+abs(F_S))
    S = F_S + F_AS*abs(F_S)/(abs(F_S)+abs(F_A)) + F_BS*abs(F_S)/(abs(F_S)+abs(F_B))+ F_ABS*abs(F_S)/(abs(F_A)+abs(F_B)+abs(F_S))
  } else {
    F_A = R2objt$Full-R2objt$B
    F_B =  R2objt$Full-R2objt$A
    F_AB = R2objt$Full - F_A -F_B
    A = F_A + F_AB*abs(F_A)/(abs(F_A)+abs(F_B))
    B = F_B + F_AB*abs(F_B)/(abs(F_A)+abs(F_B))
    S = 0
  }
  return(list(A = A, B = B, S = S, R2 = A+B+S))
}

get_null_ll = function(object) {
  if(object$family$family$family == "binomial") {
    null_m = stats::dbinom( object$data$Y, 1, 0.5, log = TRUE)
  } else if(object$family$family$family == "poisson") {
    null_m = stats::dpois( object$data$Y, 1, log = TRUE)
  } else if(object$family$family$family == "nbinom") {
    null_m = stats::dpois( object$data$Y, 1, log = TRUE)
  } else if(object$family$family$family == "gaussian") {
    null_m = stats::dnorm( object$data$Y, 0, 1.0, log = TRUE)
  }
    
  return(null_m)
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
  stats::printCoefmat(x$to_print)
  return(invisible(x$to_print))
}

#' Plot internal metacommunity structure
#' 
#' @param object anova object from \code{\link{anova.sjSDM}}
#' @param Rsquared which R squared should be used, McFadden or Nagelkerke (McFadden is default)
#' @param add_shared split shared components, default is TRUE 
#' @param env_deviance environmental deviance
#' @param suppress_plotting should the plots be suppressed or not.
#' 
#' Plots and returns the internal metacommunity structure of species and sites (see Leibold et al., 2022). 
#' Plots were heavily inspired by Leibold et al., 2022
#' 
#' @return 
#' 
#' List with the following components:
#' 
#' 
#' \item{plots}{ggplot objects for sites and species.}
#' \item{data}{List of data.frames with the internal metacommunity structure.}
#' 
#' 
#' @references 
#' Leibold, M. A., Rudolph, F. J., Blanchet, F. G., De Meester, L., Gravel, D., Hartig, F., ... & Chase, J. M. (2022). The internal structure of metacommunities. Oikos, 2022(1).
#' 
#' @export
plotInternalStructure = function(object,  
                                 Rsquared = c("McFadden", "Nagelkerke"), 
                                 add_shared = TRUE,
                                 env_deviance = NULL,
                                 suppress_plotting = FALSE) {
  Rsquared = match.arg(Rsquared)
  out = 
    plot.sjSDManova(x = object, 
                    internal = TRUE, 
                    add_shared = add_shared,
                    type = Rsquared, 
                    alpha = alpha,
                    env_deviance = env_deviance,
                    suppress_plotting = suppress_plotting)
  return(invisible(out))
}


#' Plot anova results
#' 
#' 
#' @param x anova object from \code{\link{anova.sjSDM}}
#' @param y unused argument
#' @param type deviance, Nagelkerke or McFadden R-squared
#' @param internal logical, plot internal or total structure
#' @param add_shared Add shared contributions when plotting the internal structure
#' @param cols colors for the groups
#' @param alpha alpha for colors
#' @param env_deviance environmental deviance
#' @param suppress_plotting return plots but don't plot them
#' @param ... Additional arguments to pass to \code{plot()}
#' 
#' The \code{internal = TRUE} plot was heavily inspired by Leibold et al., 2022
#' 
#' @return 
#' 
#' List with the following components:
#' 
#' If \code{internal=TRUE}:
#' 
#' \item{plots}{ggplot objects for sites and species.}
#' \item{data}{List of data.frames with the shown results.}
#' 
#' else:
#' \item{VENN}{Matrix of shown results.}
#' 
#' @references 
#' Leibold, M. A., Rudolph, F. J., Blanchet, F. G., De Meester, L., Gravel, D., Hartig, F., ... & Chase, J. M. (2022). The internal structure of metacommunities. Oikos, 2022(1).
#' 
#' @export
plot.sjSDManova = function(x, 
                           y, 
                           type = c("McFadden", "Deviance", "Nagelkerke"), 
                           internal = FALSE,
                           add_shared = TRUE,
                           cols = c("#7FC97F","#BEAED4","#FDC086"),
                           alpha=0.15, 
                           env_deviance = NULL,
                           suppress_plotting = FALSE,
                           ...) {
  lineSeq = 0.3
  nseg = 100
  dr = 1.0
  type = match.arg(type)
  out = list()
  
  oldpar = par(no.readonly = TRUE)
  on.exit(par(oldpar))
  
  if(!x$spatial && internal) {
    internal=FALSE
    warning("'internal=TRUE' currently only supported for spatial models.")
  }
  
  if(!internal) {
    
    values = x$results
    values$`R2 Nagelkerke` = ifelse(values$`R2 Nagelkerke`<0, 0, values$`R2 Nagelkerke`)
    values$`R2 McFadden` = ifelse(values$`R2 McFadden`<0, 0, values$`R2 McFadden`)
    select_rows = 
      if(x$spatial) { 
        sapply(c("F_A", "F_B", "F_AB","F_S", "F_AS", "F_BS", "F_ABS"), function(i) which(values$models == i, arr.ind = TRUE))
      } else {
        sapply(c("F_A", "F_B", "F_AB"), function(i) which(values$models == i, arr.ind = TRUE))
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
    out$VENN = values
  } else { 
  
    if(type == "Deviance") {type = "R2_McFadden"
    } else {
      if(type == "McFadden") type = "R2_McFadden"
      else type = "R2_Nagelkerke"
    }
    internals = list()
    
    if(!add_shared) {
      df = data.frame(
          env = ifelse(x$sites[[type]]$F_A<0, 0, x$sites[[type]]$F_A),
          spa = ifelse(x$sites[[type]]$F_S<0, 0, x$sites[[type]]$F_S),
          codist = ifelse(x$sites[[type]]$F_B<0, 0, x$sites[[type]]$F_B),
          r2  = ifelse(x$sites[[type]]$Full<0, 0, x$sites[[type]]$Full)/length(x$sites[[type]]$Full)
        )
      internals[[1]] = df
      names(internals)[1] = "Sites"
      
      df = data.frame(
          env = ifelse(x$species[[type]]$F_A<0, 0, x$species[[type]]$F_A),
          spa = ifelse(x$species[[type]]$F_S<0, 0, x$species[[type]]$F_S),
          codist = ifelse(x$species[[type]]$F_B<0, 0, x$species[[type]]$F_B),
          r2  = ifelse(x$species[[type]]$Full<0, 0, x$species[[type]]$Full)/length(x$species[[type]]$Full)
        )
      
      internals[[2]] = df
      names(internals)[2] = "Species"
    } else {
      type = paste0(type, "_shared")
      df = data.frame(
        env = ifelse(x$sites[[type]]$A<0, 0, x$sites[[type]]$A),
        spa = ifelse(x$sites[[type]]$S<0, 0, x$sites[[type]]$S),
        codist = ifelse(x$sites[[type]]$B<0, 0, x$sites[[type]]$B),
        r2  = ifelse(x$sites[[type]]$R2<0, 0, x$sites[[type]]$R2)/length(x$sites[[type]]$R2)
      )
      internals[[1]] = df
      names(internals)[1] = "Sites"
      
      df = data.frame(
        env = ifelse(x$species[[type]]$A<0, 0, x$species[[type]]$A),
        spa = ifelse(x$species[[type]]$S<0, 0, x$species[[type]]$S),
        codist = ifelse(x$species[[type]]$B<0, 0, x$species[[type]]$B),
        r2  = ifelse(x$species[[type]]$R2<0, 0, x$species[[type]]$R2)/length(x$species[[type]]$R2)
      )
      
      internals[[2]] = df
      names(internals)[2] = "Species"
    }
    
    
    plots_internals = list()
    
    # Code taken from https://github.com/javirudolph/iStructureMetaco/blob/master/InternalStructureMetacommunities_2021_manuscript/Figures.R
    for(i in 1:length(internals)) {
        
      add_grad = FALSE
      if((i == 1) & !is.null(env_deviance)) add_grad = TRUE
      
      top = 7
      if(i > 1) top = 1
      if(is.null(env_deviance)) top = 1
        
        r2max = ceiling(max(internals[[i]]$r2)*1e2)/1e2
        plt = 
          ggtern::ggtern(internals[[i]], ggplot2::aes_string(x = "env", z = "spa", y = "codist", size = "r2")) +
            ggtern::scale_T_continuous(limits=c(0,1),
                                       breaks=seq(0, 1,by=0.2),
                                       labels=seq(0,1, by= 0.2)) +
            ggtern::scale_L_continuous(limits=c(0,1),
                                       breaks=seq(0, 1,by=0.2),
                                       labels=seq(0, 1,by=0.2)) +
            ggtern::scale_R_continuous(limits=c(0,1),
                                       breaks=seq(0, 1,by=0.2),
                                       labels=seq(0, 1,by=0.2)) +
            #ggplot2::scale_size_area( max_size = 3) +
            ggplot2::labs(title = names(internals)[i],
                          x = "E",
                          xarrow = "Environment",
                          y = "C",
                          yarrow = "Co-Distribution",
                          z = "S", 
                          zarrow = "Spatial Autocorrelation") +
            ggtern::theme_bw() +
            ggtern::theme_showarrows() +
            ggtern::theme_arrowlong() +
            ggplot2::theme(
              panel.grid = ggplot2::element_line(color = "darkgrey", size = 0.3),
              plot.tag = ggplot2::element_text(size = 11),
              plot.title = ggplot2::element_text(size = 11, hjust = 0.1 , margin = ggplot2::margin(t = 10, b = -20)),
              tern.axis.arrow = ggplot2::element_line(size = 1),
              tern.axis.arrow.text = ggplot2::element_text(size = 6),
              axis.text = ggplot2::element_text(size = 4),
              axis.title = ggplot2::element_text(size = 6),
              legend.text = ggplot2::element_text(size = 6),
              legend.title = ggplot2::element_text(size = 8),
              strip.text = ggplot2::element_text(size = 8),
              plot.margin = unit(c(top,1,1,1)*0.2, "cm"),
            strip.background = ggplot2::element_rect(color = NA),
          ) +
          ggplot2::guides(size = ggplot2::guide_legend(title = expression(R^2), order = 1)) +
          { if(!add_grad)ggplot2::geom_point(alpha = 0.7) }+
          { if(add_grad) ggplot2::geom_point(alpha = 0.7, aes(fill=env_deviance, color = env_deviance)) }+  
          ggplot2::scale_size_continuous(range = c(0.1,5),limits = c(0, r2max), breaks = seq(0, r2max, length.out=5)) +
          { if(add_grad) ggplot2::scale_fill_gradient(low = "white", high = "black", guide = "none") } + 
          { if(add_grad) ggplot2::scale_color_gradient(low = "white", high = "black", limits = c(0, max(env_deviance))) } +
          ggplot2::theme(tern.axis.arrow.text = element_text(size = 7),legend.position = "bottom", legend.margin = margin(r = 30), legend.box="vertical") +
          { if(!add_grad) ggplot2::guides(size = ggplot2::guide_legend(title = expression(R^2), order = 1, nrow = 1, label.position = "bottom")) } +
          { if( add_grad) ggplot2::guides(size = ggplot2::guide_legend(title = expression(R^2), order = 1, nrow = 1, label.position = "bottom"),
                                          color = ggplot2::guide_colorbar(title = "Environmental deviation", title.position = "top", order = 2, barheight = 0.5, barwidth = 8)) } 
        plots_internals[[i]] = plt
      }
    if(!suppress_plotting) ggtern::grid.arrange(plots_internals[[1]], plots_internals[[2]], nrow=1, widths = c(5.0/10, 5/10))
    out$plots = plots_internals
    out$data = internals
  }
  return(invisible(out))
}

