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
#' \item{spatial}{Logical, spatial model or not.}
#' \item{species}{individual species R2s.}
#' \item{sites}{individual site R2s.}
#' \item{lls}{individual site by species negative-log-likelihood values.}
#' 
#' Implemented S3 methods are \code{\link{print.sjSDManova}} and \code{\link{plot.sjSDManova}}
#'  
#' @seealso \code{\link{plot.sjSDManova}}, \code{\link{print.sjSDManova}}
#' @import stats
#' @export

anova.sjSDM = function(object, ...) {
  out = list()
  individual = TRUE
  
  if(object$family$family$family == "gaussian") stop("gaussian not yet supported")
  
  full_m = logLik(object, individual=individual)[[1]]
  object$settings$se = FALSE
  ### fit different models ###
  #e_form = stats::as.formula(paste0(as.character(object$settings$env$formula), collapse = ""))
  if(!inherits(object, "spatial")) {
    out$spatial = FALSE
    
    
    null_m = logLik(update(object, env_formula = ~1, biotic=bioticStruct(diag = TRUE )), individual=individual)[[1]]
    A_m =    logLik(update(object, env_formula = NULL, biotic=bioticStruct(diag = TRUE )), individual=individual)[[1]]
    B_m =    logLik(update(object, env_formula = ~1, biotic=bioticStruct(diag = FALSE)), individual=individual)[[1]]
    SAT_m =  logLik(update(object, env_formula = ~as.factor(1:nrow(object$data$X)), biotic=bioticStruct(diag = TRUE )), individual=individual)[[1]]
    
    A =    A_m
    B =    B_m
    full = full_m
    AB =   null_m - ((null_m-full_m) - (null_m - A + null_m - B))
    sat =  SAT_m
    null = null_m
    anova_rows = c("Null", "B", "Full")
    names(anova_rows) = c("Null", "Biotic", "Abiotic")
    results = data.frame(models = c("A", "B","AB", "Full", "Saturated", "Null"),
                         ll = -c(sum(A), sum(B), sum(AB), sum(full), sum(sat), sum(null)))
    results_ind = list(A = -A, B = -B, AB = - AB, Full = -full, Saturated = -sat, Null = - null)
    results_predict = list(A = -A, B = -B, AB = - AB, Full = -full, Saturated = -sat, Null = - null)
    
  } else {
    out$spatial = TRUE
    s_form = stats::as.formula(paste0(as.character(object$settings$spatial$formula), collapse = ""))
    null_m = logLik(update(object, env_formula = ~1, spatial_formula= ~0, biotic=bioticStruct(diag = TRUE )), individual=individual)[[1]]
    A_m =    logLik(update(object, env_formula = NULL, spatial_formula= ~0, biotic=bioticStruct(diag = TRUE )), individual=individual)[[1]]
    B_m =    logLik(update(object, env_formula = ~1, spatial_formula= ~0, biotic=bioticStruct(diag = FALSE)), individual=individual)[[1]]
    S_m =    logLik(update(object, env_formula = ~1, spatial_formula= NULL, biotic=bioticStruct(diag = TRUE)), individual=individual)[[1]]
    AB_m =   logLik(update(object, env_formula = NULL, spatial_formula= ~0, biotic=bioticStruct(diag = FALSE )), individual=individual)[[1]]
    AS_m =   logLik(update(object, env_formula = NULL, spatial_formula= NULL, biotic=bioticStruct(diag = FALSE )), individual=individual)[[1]]
    BS_m =   logLik(update(object, env_formula = ~1, spatial_formula= NULL, biotic=bioticStruct(diag = TRUE )), individual=individual)[[1]]
    SAT_m =  logLik(update(object, env_formula = ~as.factor(1:nrow(object$data$X)), spatial_formula= ~0, biotic=bioticStruct(diag = TRUE )), individual=individual)[[1]]
    
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
                         ll = -c(sum(A), sum(B),sum(S),sum(AB_m),sum(BS_m), sum(AS_m), sum(AB), sum(AS), sum(BS), sum(ABS), sum(full), sum(sat), sum(null)))
    
    results_ind = list("A"=-A, "B"=-B,"S"=-S,"B+A"=-AB_m,"B+S"=-BS_m,"A+S"=-AS_m,"AB"=-AB,"AS"=-AS, "BS"=-BS, "ABS"=-ABS, "Full"=-full, "Saturated"=-sat, "Null"=-null)
    
    anova_rows = c("Null", "B", "B+A", "Full")
    names(anova_rows) = c("Null", "Biotic", "Abiotic", "Spatial")
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
  out$species = list(Residual_deviance = Residual_deviance_ind,
                     Deviance = Deviance_ind,
                     R2_Nagelkerke = R2_Nagelkerke_ind,
                     R2_McFadden = R2_McFadden_ind)
  out$sites = list(R2_Nagelkerke = R2_Nagelkerke_sites,
                   R2_McFadden = R2_McFadden_sites)
  out$lls = list(results_ind)
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
#' @param type deviance, Nagelkerke or McFadden R-squared
#' @param internal logical, plot internal or total structure
#' @param cols colors for the groups
#' @param alpha alpha for colors
#' @param env_deviance environmental deviance
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
                           type = c("Deviance", "Nagelkerke", "McFadden"), 
                           internal = FALSE,
                           cols = c("#7FC97F","#BEAED4","#FDC086"),
                           alpha=0.15, 
                           env_deviance = NULL,
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
    out$VENN = values
  } else { 
  
    if(type == "Deviance") {type = "R2_McFadden"
    } else {
      if(type == "McFadden") type = "R2_McFadden"
      else type = "R2_Nagelkerke"
    }
    internals = list()
    

    df = data.frame(
        env = ifelse(x$sites[[type]]$A<0, 0, x$sites[[type]]$A),
        spa = ifelse(x$sites[[type]]$S<0, 0, x$sites[[type]]$S),
        codist = ifelse(x$sites[[type]]$B<0, 0, x$sites[[type]]$B),
        r2  = ifelse(x$sites[[type]]$Full<0, 0, x$sites[[type]]$Full)/length(x$sites[[type]]$Full)
      )
    internals[[1]] = df
    names(internals)[1] = "Sites"
    
    df = data.frame(
        env = ifelse(x$species[[type]]$A<0, 0, x$species[[type]]$A),
        spa = ifelse(x$species[[type]]$S<0, 0, x$species[[type]]$S),
        codist = ifelse(x$species[[type]]$B<0, 0, x$species[[type]]$B),
        r2  = ifelse(x$species[[type]]$Full<0, 0, x$species[[type]]$Full)/length(x$species[[type]]$Full)
      )
    
    internals[[2]] = df
    names(internals)[2] = "Species"
      
    
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
    ggtern::grid.arrange(plots_internals[[1]], plots_internals[[2]], nrow=1, widths = c(5.0/10, 5/10))
    out$plots = plots_internals
    out$data = internals
  }
  return(invisible(out))
}

