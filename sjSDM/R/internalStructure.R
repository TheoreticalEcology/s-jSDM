

#' Plot internal metacommunity structure
#' 
#' @param object anova object from \code{\link{anova.sjSDM}}
#' @param Rsquared which R squared should be used, McFadden or Nagelkerke (McFadden is default)
#' @param fractions how to handle shared fractions
#' @param negatives how to handle negative R squareds
#' @param plot should the plots be suppressed or not.
#' 
#' Plots and returns the internal metacommunity structure of species and sites (see Leibold et al., 2022). 
#' Plots were heavily inspired by Leibold et al., 2022
#' 
#' @return 
#' 
#' An object of class sjSDMinternalStructure consisting of a list of data.frames with the internal structure. 
#' 
#' @seealso [plot.sjSDMinternalStructure], [print.sjSDMinternalStructure]
#' 
#' @example /inst/examples/anova-example.R
#' @references 
#' Leibold, M. A., Rudolph, F. J., Blanchet, F. G., De Meester, L., Gravel, D., Hartig, F., ... & Chase, J. M. (2022). The internal structure of metacommunities. Oikos, 2022(1).
#' 
#' @export
internalStructure = function(object,  
                             Rsquared = c("McFadden", "Nagelkerke"), 
                             fractions = c("discard", "proportional", "equal"),
                             negatives = c("floor", "scale", "raw"), # TODO - rounding ANOVA out, here all calculations to function with option
                             plot = FALSE) {
  
  fractions = match.arg(fractions)
  Rsquared = match.arg(Rsquared)
  negatives = match.arg(negatives)
  
  if(!object$spatial) stop("'internal structure' currently only supported for spatial models.")  
  
  if(Rsquared == "Deviance") {type = "R2_McFadden"
  } else {
    if(Rsquared == "McFadden") type = "R2_McFadden"
    else type = "R2_Nagelkerke"
  }
  
  internals = list()
  
  if(fractions == "discard") {
    df = data.frame(
      env = object$sites[[type]]$F_A,
      spa = object$sites[[type]]$F_S,
      codist = object$sites[[type]]$F_B,
      r2  = object$sites[[type]]$Full #/length(object$sites[[type]]$Ful)
    )
    
    
    internals[[1]] = df
    names(internals)[1] = "Sites"
    
    df = data.frame(
      env = object$species[[type]]$F_A,
      spa = object$species[[type]]$F_S,
      codist = object$species[[type]]$F_B,
      r2  = object$species[[type]]$Full
    )
    
    internals[[2]] = df
    names(internals)[2] = "Species"
  } else {
    type = paste0(type, "_shared")
    if(fractions == "proportional") {
      
      df = data.frame(
        env = object$sites[[type]]$proportional$F_A,
        spa = object$sites[[type]]$proportional$F_S,
        codist = object$sites[[type]]$proportional$F_B,
        r2  = object$sites[[type]]$proportional$R2 #/length(object$sites[[type]]$R2)
      )
      
      internals[[1]] = df
      names(internals)[1] = "Sites"
      
      df = data.frame(
        env = object$species[[type]]$proportional$F_A,
        spa = object$species[[type]]$proportional$F_S,
        codist = object$species[[type]]$proportional$F_B,
        r2  = object$species[[type]]$proportional$R2 #/length(object$species[[type]]$R2)
      )
      
      
      internals[[2]] = df
      names(internals)[2] = "Species"
      
    } else {
      
      df = data.frame(
        env = object$sites[[type]]$equal$F_A ,
        spa = object$sites[[type]]$equal$F_S ,
        codist = object$sites[[type]]$equal$F_B ,
        r2  = object$sites[[type]]$equal$R2 #/length(object$sites[[type]]$R2)
      )
      

      internals[[1]] = df
      names(internals)[1] = "Sites"
      
      df = data.frame(
        env = object$species[[type]]$equal$F_A,
        spa = object$species[[type]]$equal$F_S,
        codist = object$species[[type]]$equal$F_B,
        r2  = object$species[[type]]$equal$R2 #/length(object$species[[type]]$R2)
      )
      
      
      internals[[2]] = df
      names(internals)[2] = "Species"
    }
      
    }
    
  out = list()
  out$raws = internals
  internals[[1]] = standardize_df(internals[[1]], standardize = negatives )
  internals[[2]] = standardize_df(internals[[2]], standardize = negatives )
  out$internals = internals
  out$Rsquared = Rsquared
  out$fractions = fractions
  out$anova = object
  class(out) = "sjSDMinternalStructure"
  
  if(plot == T) plot(out)
  
  return(out)
}


standardize_df = function(df, standardize) {
  if(standardize == "floor") {
    tmp = df[,1:3]
    tmp[tmp<0] = 0
    df[,1:3] = tmp
  } else if (standardize == "abs") {
    df[,1:3] = abs(df[,1:3])
  } else if(standardize == "scale" ){
    tmp = df[,1:3]
    tmp = scales::rescale(as.matrix(tmp), to = c(0, 1))
    df[,1:3] = tmp[,1:3]
  } 
  return(df)
}


#' Print internal structure object
#' 
#' @param x object of class sjSDMinternalStructure
#' @param ... no function
#' 
#' @export
print.sjSDMinternalStructure <- function(x, ...){
  return(x$internals)
}


#' Plot internal structure
#' 
#' Creates a ternary diagram of an object of class 
#' 
#' @param x and object of class sjSDMinternalStructure create by anova object from \code{\link{internalStructure}}
#' @param alpha alpha of points
#' @param env_deviance environmental deviance/gradient (points will be colored)
#' @param negatives how to handle negative R squareds
#' @param ... no function
#' 
#' 
#' @example /inst/examples/anova-example.R
#' @export
#' 
plot.sjSDMinternalStructure <- function(x, 
                                       alpha = 0.15,
                                       env_deviance = NULL,
                                       negatives = c("floor", "scale", "raw"),
                                       ...){
  
  negatives = match.arg(negatives)
  internals = x$raws
  internals[[1]] = standardize_df(internals[[1]], negatives)
  internals[[2]] = standardize_df(internals[[2]], negatives)
  plots_internals = list()
  
  old_par = par(no.readonly = TRUE)
  on.exit(par(old_par))
  
  # Code taken from https://github.com/javirudolph/iStructureMetaco/blob/master/InternalStructureMetacommunities_2021_manuscript/Figures.R
  
  
  if(min(internals[[1]][,1:3]) < 0 | min(internals[[2]][,1:3]) < 0) {
    message("Negative partial R-square detected. Negative R-squareds can occur but they cannot be displayed by the ternary plot.")
  }
  par(mfrow = c(1,2))
  
  for(i in 1:length(internals)) {
    
    

    
    
    add_grad = FALSE
    if((i == 1) & !is.null(env_deviance)) add_grad = TRUE
    
    top = 7
    if(i > 1) top = 1
    if(is.null(env_deviance)) top = 1
    
    negative_r2 = FALSE
    if(min(internals[[i]]$r2) < 0) negative_r2 = TRUE
    
    r2max = ceiling(max(internals[[i]]$r2)*1e2)/1e2
    r2min = floor(min(internals[[i]]$r2)*1e2)/1e2
    
    color = if(!negative_r2) {NULL} else {"r2"}
    

    min_s = c(internals[[i]]$r2) |> abs() |> min()
    max_s = c(internals[[i]]$r2) |> abs() |> max()
    labels_sizes = round(seq(min_s, max_s, length.out = 4), 3)
    sizes_legend = scales::rescale(labels_sizes, to = c(0.2, 2.0) )

    plot(NULL, NULL, xlim = c(0, 1), ylim = c(0, 1), axes=F, xlab ="", ylab = '')
    

    if(!negative_r2) {
      points(x = seq(-0.05, 0.25, length.out = 4), y = rep(0.86, 4),cex = sizes_legend, xpd = NA, pch = 16)
      text(x = seq(-0.05, 0.25, length.out = 4), y = rep(0.852, 4), labels = labels_sizes, xpd = NA, pos = 1, cex = 0.7)
      text(x = -0.1, y = 0.89, labels = "R\u00B2:", xpd = NA, cex = 0.9)
      cols = "#000000AA"
    } else {
      cols1 =  grDevices::colorRampPalette(c("#FF1010", "#FFA4B5"))(10)
      cols2 =  grDevices::colorRampPalette(c("grey70", "black"))(10)
      image(x=seq(-0.05, 0.2, length.out = 20), y=c(0.86, 0.885), z=matrix(seq(-0.05, 0.2, length.out=20), ncol=1),
            col=c(cols1, cols2), axes=FALSE, xlab="", ylab="", add = TRUE)
      r2s = internals[[i]]$r2
      text(x = c(-0.05, 0.075, 0.2), y = rep(0.852, 3), labels = c(min(r2s) |> round(3), 0, max(r2s) |> round(3)), xpd = NA, pos = 1, cex = 0.7)
      text(x = -0.1, y = 0.89, labels = "R\u00B2:", xpd = NA, cex = 0.9)
      names(r2s)[r2s<0] = cols1[as.integer(cut(r2s[r2s<0], breaks = 10))]
      names(r2s)[r2s>0] = cols2[as.integer(cut(r2s[r2s>0], breaks = 10))]
      cols = names(r2s)
    }
    
    if(add_grad) {
      br = min(length(unique(env_deviance)), 20)
      bg = viridis::viridis(br)[cut(env_deviance, breaks = br)]
      
      image(x=seq(-0.05, 0.2, length.out = 20)+0.8, y=c(0.86, 0.885), z=matrix(seq(-0.05, 0.2, length.out=20), ncol=1),
            col=viridis::viridis(20), axes=FALSE, xlab="", ylab="", add = TRUE)
      text(x = c(-0.05, 0.2)+0.8, y = rep(0.852, 2), labels = c(min(env_deviance) |> round(3),max(env_deviance) |> round(3)), xpd = NA, pos = 1, cex = 0.7)
    } else {
      bg = "#000000AA"
    }
    
    plot_tern(internals[[i]], cex = scales::rescale(abs(internals[[i]]$r2), to = c(0.4, 2.0), from = c(min_s, max_s) ), col = cols, bg = bg)
    
    if(i == 1) {
      text(x = -0.1, y = 1.0, label = "Sites", font = 2, xpd = NA)
    } else {
      text(x = -0.1, y = 1.0, label = "Species", font = 2, xpd = NA)
    }
    
    
  #   plt = 
  #     ggtern::ggtern(internals[[i]], ggplot2::aes_string(x = "env", z = "spa", y = "codist", size = abs(internals[[i]]$r2), color = color) ) +
  #     ggtern::scale_T_continuous(limits=c(0,1),
  #                                breaks=seq(0, 1,by=0.2),
  #                                labels=seq(0,1, by= 0.2)) +
  #     ggtern::scale_L_continuous(limits=c(0,1),
  #                                breaks=seq(0, 1,by=0.2),
  #                                labels=seq(0, 1,by=0.2)) +
  #     ggtern::scale_R_continuous(limits=c(0,1),
  #                                breaks=seq(0, 1,by=0.2),
  #                                labels=seq(0, 1,by=0.2)) +
  #     #ggplot2::scale_size_area( max_size = 3) +
  #     ggplot2::labs(title = names(internals)[i],
  #                   x = "E",
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
  #       plot.margin = unit(c(top,1,1,1)*0.2, "cm"),
  #       strip.background = ggplot2::element_rect(color = NA),
  #     ) +
  #     { if(!negative_r2) ggplot2::guides(size = ggplot2::guide_legend(title = expression(R^2), order = 1)) } +
  #     { if(!add_grad)ggplot2::geom_point(alpha = 0.7) }+
  #     { if(add_grad) ggplot2::geom_point(alpha = 0.7, aes(fill=env_deviance, color = env_deviance)) }+  
  #     { if(!negative_r2)  ggplot2::scale_size_continuous(range = c(0.1,5),limits = c(r2min, r2max), breaks = seq(r2min, r2max, length.out=5)) } +
  #     { if(negative_r2)   ggplot2::scale_color_gradient2(low = "red", mid = "grey50", high = "black", midpoint = 0, 
  #                                                 breaks = c(r2min, 0, r2max), 
  #                                                 labels = c(r2min, "0", r2max), 
  #                                                 limits = c(r2min, r2max),
  #                                                 guide = ggplot2::guide_colorbar(title = expression(R^2))) } +
  #     { if(negative_r2)  ggplot2::scale_size_continuous(range = c(0.1,5), 
  #                                              breaks =  seq(0, r2max, length.out = 5), guide = "none")  } +
  #     
  #     { if(add_grad) ggplot2::scale_fill_gradient(low = "white", high = "black", guide = "none") } + 
  #     { if(add_grad) ggplot2::scale_color_gradient(low = "white", high = "black", limits = c(0, max(env_deviance))) } +
  #     ggplot2::theme(tern.axis.arrow.text = element_text(size = 7),legend.position = "bottom", legend.margin = margin(r = 30), legend.box="vertical") +
  #     { if(!add_grad) { if(!negative_r2)  ggplot2::guides(size = ggplot2::guide_legend(title = expression(R^2), order = 1, nrow = 1, label.position = "bottom")) } } +
  #     { if( add_grad) ggplot2::guides(size = ggplot2::guide_legend(title = expression(R^2), order = 1, nrow = 1, label.position = "bottom"),
  #                                     color = ggplot2::guide_colorbar(title = "Environmental deviation", title.position = "top", order = 2, barheight = 0.5, barwidth = 8)) } 
  #   plots_internals[[i]] = plt
   }
  # 
  # ggtern::grid.arrange(plots_internals[[1]], plots_internals[[2]], nrow=1, widths = c(5.0/10, 5/10))
  
  out = list()
  #out$plots = plots_internals
  return(invisible(out))
}

#' Plot predictors of assembly processes 
#' 
#' The function plots correlations between assembly processes and predictors or traits
#' 
#' @param object An \code{sjSDManova} object from the \code{\link{anova.sjSDM}} function.
#' @param response whether to use sites or species. Default is sites
#' @param pred predictor variable. If \code{NULL}, environment uniqueness, spatial uniqueness, and richness is calculated from the fitted object and used as predictor. 
#' @param cols Colors for the three assembly processes.
#' @param negatives how to handle negative R squareds
#' 
#' @details
#'  
#' Correlation and plots of the three assembly processes (environment, space, and codist) against environmental and spatial uniqueness and richness. The importance of the three assembly processes is measured by the partial R-squared (shown in the internal structure plots).
#' 
#' Importances are available for species and sites. Custom environmental predictors or traits can be specified. Environmental predictors are plotted against site R-squared and traits are plotted against species R-squared.
#' Regression lines are estimated by 50\% quantile regression models.
#' 
#' @note Defaults for negative values are different than for [plot.sjSDMinternalStructure]
#' 
#' 
#' @return
#' 
#' A list with the following components:
#'
#' \item{env}{A list of summary tables for env, space, and codist R-squared.}
#' \item{space}{A list of summary tables for env, space, and codist R-squared.}
#' \item{codist}{A list of summary tables for env, space, and codist R-squared.}
#' 
#' @references
#' 
#' Leibold, M. A., Rudolph, F. J., Blanchet, F. G., De Meester, L., Gravel, D., Hartig, F., ... & Chase, J. M. (2022). The internal structure of metacommunities. *Oikos*, 2022(1).
#' 
#' @example /inst/examples/anova-example.R
#' @export
plotAssemblyEffects = function(object, 
                               response = c("sites", "species"),
                               pred = NULL,
                               cols = c("#A38310", "#B42398", "#20A382"),
                               negatives = c("raw", "scale", "floor")
                               ) {
  
  response = match.arg(response)
  if (response == "species" & is.null(pred)) stop("Species response requires predictors")
  
  oldpar = par(no.readonly = TRUE)
  on.exit(par(oldpar))
  
  negatives = match.arg(negatives)
  object$internals$Sites = standardize_df(object$raws$Sites, negatives)
  object$internals$Species = standardize_df(object$raws$Species, negatives)

  lwd = 2
  
  X = object$anova$object$settings$env$X
  XYcoords = object$anova$object$settings$spatial$X
  Y = object$anova$object$data$Y
  rr = object
  minR = min(rr$internals$Sites[,1:3])
  maxR = max(rr$internals$Sites[,1:3])
  minRS = min(rr$internals$Species[,1:3])
  maxRS = max(rr$internals$Species[,1:3])
  
  out = list()
  
  if(is.null(pred)) {
    
    graphics::par(mfrow = c(1, 3), mar = c(4, 4, 4, 1), xaxt= "s")
    env_eigen = get_eigen(scale(X), FALSE)
    spatial_eigen = get_eigen(scale(XYcoords),FALSE)
    richness = rowSums(Y)
    
    env_eigen_centered = scale(env_eigen, center = TRUE, scale = FALSE)
    spatial_eigen_centered = scale(spatial_eigen, center = TRUE, scale = FALSE)
    richness_centered = scale(richness, center = TRUE, scale = FALSE)
    
    
    out$env = list()
    out$space = list()
    out$codist = list()
    

    
    graphics::plot(NULL, NULL, xlim = c(min(env_eigen_centered), max(env_eigen_centered)), 
                   ylim = c(minR, maxR), xlab = "Env uniqueness",main = "", ylab = "R2", las =1)
    
    for(i in 1:3) {
      graphics::points(env_eigen_centered, rr$internals$Sites[,i], col = ggplot2::alpha(cols[i], 0.2), pch = 16)
      g = qgam::qgam( Y ~ env_eigen_centered + spatial_eigen_centered + richness_centered, 
                      data = data.frame(Y = rr$internals$Sites[,i], 
                                        env_eigen_centered = env_eigen_centered, 
                                        spatial_eigen_centered = spatial_eigen_centered, 
                                        richness_centered = richness_centered), 
                      qu = 0.5, control = list(progress="none"))
      out$env[[colnames(rr$internals$Sites)[i]]] = g 
      graphics::abline(a = coef(g)[c(1, 2)], col = cols[i], lwd = lwd, lty =  1*(summary(g)$p.table[2,4] > 0.05)+1 )
      
    }
    graphics::legend("topright", legend = c("env", "spa", "codist"), col = cols, pch = 15, bty = "n")
    graphics::legend("topleft", legend = c("significant", "non-significant"), col = c("black", "black"),  bty = "n", lty = c(1,2))
    
    graphics::plot(NULL, NULL, xlim = c(min(spatial_eigen_centered), max(spatial_eigen_centered)), ylim = c(minR, maxR), xlab = "Spatial uniqueness",main = "", ylab = "R2", las =1)
    for(i in 1:3) {
      graphics::points(spatial_eigen_centered, rr$internals$Sites[,i], col = ggplot2::alpha(cols[i], 0.2), pch = 16)
      g = qgam::qgam( Y ~ env_eigen_centered + spatial_eigen_centered + richness_centered, 
                      data = data.frame(Y = rr$internals$Sites[,i], 
                                        env_eigen_centered = env_eigen_centered, 
                                        spatial_eigen_centered = spatial_eigen_centered, 
                                        richness_centered = richness_centered), 
                      qu = 0.5, control = list(progress="none"))
      out$space[[colnames(rr$internals$Sites)[i]]] = g 
      graphics::abline(a = coef(g)[c(1, 3)], col = cols[i], lwd = lwd, lty =  1*(summary(g)$p.table[3,4] > 0.05)+1 )
      
    }
    graphics::legend("topright", legend = c("env", "spa", "codist"), col = cols, pch = 15, bty = "n")
    graphics::legend("topleft", legend = c("significant", "non-significant"), col = c("black", "black"),  bty = "n", lty = c(1,2))
    
    
    graphics::plot(NULL, NULL, xlim = c(min(richness_centered), max(richness_centered)), ylim = c(minR, maxR), xlab = "Richness",main = "", ylab = "R2", las =1)
    for(i in 1:3) {
      graphics::points(richness_centered, rr$internals$Sites[,i], col = ggplot2::alpha(cols[i], 0.2), pch = 16)
      g = qgam::qgam( Y ~ env_eigen_centered + spatial_eigen_centered + richness_centered, 
                      data = data.frame(Y = rr$internals$Sites[,i], 
                                        env_eigen_centered = env_eigen_centered, 
                                        spatial_eigen_centered = spatial_eigen_centered, 
                                        richness_centered = richness_centered), 
                      qu = 0.5, control = list(progress="none"))
      out$codist[[colnames(rr$internals$Sites)[i]]] = g 
      graphics::abline(a = coef(g)[c(1, 4)], col = cols[i], lwd = lwd, lty =  1*(summary(g)$p.table[4,4] > 0.05)+1 )
      
    }
    graphics::legend("topright", legend = c("env", "spa", "codist"), col = cols, pch = 15, bty = "n")
    graphics::legend("topleft", legend = c("significant", "non-significant"), col = c("black", "black"),  bty = "n", lty = c(1,2))
  } else {
    if(response == "sites") {
      if(is.factor(pred) || is.character(pred)) {
        group = pred
        df = data.frame(R2 = c(rr$internals$Sites$env, rr$internals$Sites$spa, rr$internals$Sites$codist),
                        part = rep(c("env", "spa", "codist"), each = nrow(rr$internals$Sites)),
                        group = rep(group, 3)
        )
        b = graphics::boxplot(R2~part+group, data =df,las = 2, col = alpha(cols[c(3, 1, 2)],0.7), xlab = "", main = "", notch = TRUE )
        b = beeswarm::beeswarm(R2~part+group, data =df,las = 2, col = cols[c(3, 1, 2)], xlab = "", main = "" , add=TRUE, method = "center", spacing = 0.3)
        graphics::legend("topright", legend = c("env", "spa", "codist"), col = cols, pch = 15, bty = "n")
      } else {
        graphics::par(mfrow = c(1, 1), mar = c(4, 4, 4, 1), xaxt= "s")
        sPred = scale(pred, center = TRUE, scale = FALSE)
        
        graphics::plot(NULL, NULL, xlim = c(min(sPred), max(sPred)), ylim = c(minR, maxR), xlab = "sPredictor",main = "", ylab = "R2", las =1)
        for(i in 1:3) {
          graphics::points(sPred, rr$internals$Sites[,i], col = ggplot2::alpha(cols[i], 0.2), pch = 16)
          g = qgam::qgam( Y ~ sPred, data = data.frame(Y = rr$internals$Sites[,i], sPred = sPred), qu = 0.5, control = list(progress="none"))
          graphics::abline(a = coef(g)[c(1, 2)], col = cols[i], lwd = lwd, lty =  1*(summary(g)$p.table[2,4] > 0.05)+1 )
          out$sPred[[colnames(rr$internals$Sites)[i]]] = g 
        }
        graphics::legend("topright", legend = c("env", "spa", "codist"), col = cols, pch = 15, bty = "n")
        graphics::legend("topleft", legend = c("significant", "non-significant"), col = c("black", "black"),  bty = "n", lty = c(1,2))
      }
    }
    
    if(response == "species") {
      if(is.factor(pred) || is.character(pred)) {
        group = pred
        df = data.frame(R2 = c(rr$internals$Species$env, rr$internals$Species$spa, rr$internals$Species$codist),
                        part = rep(c("env", "spa", "codist"), each = nrow(rr$internals$Species)),
                        group = rep(group, 3)
        )
        b = graphics::boxplot(R2~part+group, data =df,las = 2, col = alpha(cols[c(3, 1, 2)],0.7), xlab = "", main = "", notch = TRUE )
        b = beeswarm::beeswarm(R2~part+group, data =df,las = 2, col = cols[c(3, 1, 2)], xlab = "", main = "" , add=TRUE, method = "center", spacing = 0.3)
        graphics::legend("topright", legend = c("env", "spa", "codist"), col = cols, pch = 15, bty = "n")
      } else {
        graphics::par(mfrow = c(1, 1), mar = c(4, 4, 4, 1), xaxt= "s")
        sPred = scale(pred, center = TRUE, scale = FALSE)
        
        graphics::plot(NULL, NULL, xlim = c(min(sPred), max(sPred)), ylim = c(minRS, maxRS), xlab = "sPredictor",main = "", ylab = "R2", las =1)
        for(i in 1:3) {
          graphics::points(sPred, rr$internals$Species[,i], col = ggplot2::alpha(cols[i], 0.2), pch = 16)
          g = qgam::qgam( Y ~ sPred, data = data.frame(Y = rr$internals$Species[,i], sPred = sPred), qu = 0.5, control = list(progress="none"))
          graphics::abline(a = coef(g)[c(1, 2)], col = cols[i], lwd = lwd, lty =  1*(summary(g)$p.table[2,4] > 0.05)+1 )
          out$sPred[[colnames(rr$internals$Species)[i]]] = g 
        }
        graphics::legend("topright", legend = c("env", "spa", "codist"), col = cols, pch = 15, bty = "n")
        graphics::legend("topleft", legend = c("significant", "non-significant"), col = c("black", "black"),  bty = "n", lty = c(1,2))
      }
      
    }
    
    
  }
  
  return(invisible(out))
}




get_eigen = function(X, double_center = TRUE, full = FALSE) {
  D = as.matrix(dist(scale(X)))
  if(double_center) D = D - (diag(0.0, ncol(D)) + rowMeans(D)) - t(diag(0.0, ncol(D)) + colMeans(D)) + mean(D) # Double center
  eig = eigen(D)
  if(!full)return(abs(eig$vectors[,which.max(abs(eig$values))]))
  else {
    return(list(x = eig$vectors))
  }
}










rad2deg <- function(rad) {(rad * 180) / (pi)}
deg2rad <- function(deg) {(deg * pi) / (180)}
get_coords = function(X) {
  x_w = X[1] + sin(deg2rad(30))*X[2]
  y_w = cos(deg2rad(30))*X[2]
  return(c(x_w, y_w))
}



plot_line = function(X, Y, arrow= FALSE,...) {
  Xc = get_coords(X)
  Yc = get_coords(Y)
  
  if(arrow) {
    arrows(Xc[1], Xc[2], Yc[1], Yc[2],...)
  } else {
    segments(Xc[1], Xc[2], Yc[1], Yc[2],...)
  }
}
plot_tern = function(data1, length = 0.12, col = "black", bg = NULL, cex = 1.0) {
  segments(0,0,1,0)
  segments(0, 0.0, 0.5, 0.8660254)
  segments(1,0,0.5, 0.8660254)
  for(i in seq(0.2, 0.8, length.out = 4)) {
    plot_line(c(1-i, 0.0, i), c(0.0, 1-i, i), col = "lightgrey")
    plot_line(c(1-i, i, 0.0), c(1-i, 0.0, i), col = "lightgrey")
    plot_line(c(0, i, 1-i), c(1-i, i, 0), col = "lightgrey")
    
    text(get_coords(c(1-i, 0.0, i))[1], get_coords(c(1-i, 0.0, i))[2]-0.03, labels = 1-i, srt = 60, xpd = NA)
    text(get_coords(c(1-i, i, 0.0))[1]+0.05, get_coords(c(1-i, i, 0.0))[2], labels = i, xpd = NA)
    text(get_coords(c(0, i, 1-i))[1]-0.03, get_coords(c(0, i, 1-i))[2]+0.03, labels = 1-i, xpd = NA, srt  = -50)
  }
  #arrows(0.2,-0.05,0.8,-0.05, xpd = NA,length = 0.1)
  text(1, y = -0.02, pos = 1, xpd = NA, label = "Space")
  text(-0.1, y = -0.02, pos = 1, xpd = NA, label = "Environment")
  text(0.5, y = 0.9, pos = 3, xpd = NA, label = "Species associations")
  

  
  data1[,1:3] = data1[,1:3]/rowSums(data1[,1:3])
  if(length(col) == 1) col = rep(col, nrow(data1))
  if(length(cex) == 1) cex = rep(cex, nrow(data1))
  if(is.null(bg)) bg = col
  if(length(bg) == 1) bg = rep(bg, nrow(data1))
  for(i in 1:nrow(data1)) {
    coords = get_coords(unlist(data1[i,c(2, 3, 1)]))
    points(x = coords[1], y = coords[2], col = col[i], bg = bg[i], cex = cex[i], pch = 21)
  }
}

