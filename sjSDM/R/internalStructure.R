

#' Plot internal metacommunity structure
#' 
#' @param object anova object from \code{\link{anova.sjSDM}}
#' @param Rsquared which R squared should be used, McFadden or Nagelkerke (McFadden is default)
#' @param fractions how to handle shared fractions
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
internalStructure = function(object,  
                             Rsquared = c("McFadden", "Nagelkerke"), 
                             fractions = c("discard", "proportional", "equal"),
                             standardize = c("scale", "floor"), # TODO - rounding ANOVA out, here all calculations to function with option
                             plot = FALSE) {
  
  fractions = match.arg(fractions)
  Rsquared = match.arg(Rsquared)
  standardize = match.arg(standardize)
  
  if(!object$spatial) warning("'internal=TRUE' currently only supported for spatial models.")  
  
  if(Rsquared == "Deviance") {type = "R2_McFadden"
  } else {
    if(Rsquared == "McFadden") type = "R2_McFadden"
    else type = "R2_Nagelkerke"
  }
  
  internals = list()
  
  if(fractions == "discard") {
    df = data.frame(
      env = ifelse(object$sites[[type]]$F_A<0, 0, object$sites[[type]]$F_A),
      spa = ifelse(object$sites[[type]]$F_S<0, 0, object$sites[[type]]$F_S),
      codist = ifelse(object$sites[[type]]$F_B<0, 0, object$sites[[type]]$F_B),
      r2  = ifelse(object$sites[[type]]$Full<0, 0, object$sites[[type]]$Full)#/length(object$sites[[type]]$Full)
    )
    internals[[1]] = df
    names(internals)[1] = "Sites"
    
    df = data.frame(
      env = ifelse(object$species[[type]]$F_A<0, 0, object$species[[type]]$F_A),
      spa = ifelse(object$species[[type]]$F_S<0, 0, object$species[[type]]$F_S),
      codist = ifelse(object$species[[type]]$F_B<0, 0, object$species[[type]]$F_B),
      r2  = ifelse(object$species[[type]]$Full<0, 0, object$species[[type]]$Full)#/length(object$species[[type]]$Full)
    )
    
    internals[[2]] = df
    names(internals)[2] = "Species"
  } else {
    type = paste0(type, "_shared")
    if(fractions == "proportional") {
      
      df = data.frame(
        env = ifelse(object$sites[[type]]$proportional$F_A<0, 0, object$sites[[type]]$proportional$F_A),
        spa = ifelse(object$sites[[type]]$proportional$F_S<0, 0, object$sites[[type]]$proportional$F_S),
        codist = ifelse(object$sites[[type]]$proportional$F_B<0, 0, object$sites[[type]]$proportional$F_B),
        r2  = ifelse(object$sites[[type]]$proportional$R2<0, 0, object$sites[[type]]$proportional$R2)#/length(object$sites[[type]]$R2)
      )
      internals[[1]] = df
      names(internals)[1] = "Sites"
      
      df = data.frame(
        env = ifelse(object$species[[type]]$proportional$F_A<0, 0, object$species[[type]]$proportional$F_A),
        spa = ifelse(object$species[[type]]$proportional$F_S<0, 0, object$species[[type]]$proportional$F_S),
        codist = ifelse(object$species[[type]]$proportional$F_B<0, 0, object$species[[type]]$proportional$F_B),
        r2  = ifelse(object$species[[type]]$proportional$R2<0, 0, object$species[[type]]$proportional$R2)#/length(object$species[[type]]$R2)
      )
      
    } else {
      
      df = data.frame(
        env = ifelse(object$sites[[type]]$equal$F_A<0, 0, object$sites[[type]]$equal$F_A),
        spa = ifelse(object$sites[[type]]$equal$F_S<0, 0, object$sites[[type]]$equal$F_S),
        codist = ifelse(object$sites[[type]]$equal$F_B<0, 0, object$sites[[type]]$equal$F_B),
        r2  = ifelse(object$sites[[type]]$equal$R2<0, 0, object$sites[[type]]$equal$R2)#/length(object$sites[[type]]$R2)
      )
      internals[[1]] = df
      names(internals)[1] = "Sites"
      
      df = data.frame(
        env = ifelse(object$species[[type]]$equal$F_A<0, 0, object$species[[type]]$equal$F_A),
        spa = ifelse(object$species[[type]]$equal$F_S<0, 0, object$species[[type]]$equal$F_S),
        codist = ifelse(object$species[[type]]$equal$F_B<0, 0, object$species[[type]]$equal$F_B),
        r2  = ifelse(object$species[[type]]$equal$R2<0, 0, object$species[[type]]$equal$R2)#/length(object$species[[type]]$R2)
      )
      
    }
    
    internals[[2]] = df
    names(internals)[2] = "Species"
  }
  
  out = list()
  out$internals = internals
  out$Rsquared = Rsquared
  out$fractions = fractions
  out$anova = object
  class(out) = "sjSDMinternalStruture"
  
  if(plot == T) plot(out)
  
  return(out)
}


#' Print internal structure object
#' 
#' @param x object of class sjSDMinternalStruture
#' @param ... no function
#' 
#' @export
print.sjSDMinternalStruture <- function(x, ...){
  return(x$internals)
}


#' Plot internal structure
#' 
#' Creates a ternary diagram of an object of class 
#' 
#' @param object and object of class sjSDMinternalStruture create by anova object from \code{\link{internalStructure}}
#' 
#' @export
#' 
plot.sjSDMinternalStruture <- function(x, 
                                       alpha = 0.15,
                                       env_deviance = NULL,
                                       ...){
  
  internals = x$internals
  
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
      ggtern::ggtern(internals[[i]], ggplot2::aes_string(x = "env", z = "spa", y = "codist", size = "r2"))+
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
                    yarrow = "Species associations",
                    z = "S", 
                    zarrow = "Space") +
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
  
  out = list()
  out$plots = plots_internals
  return(invisible(out))
}

#' Plot Correlations between assembly processes and predictors or traits
#' 
#' @param object An \code{sjSDManova} object from the \code{\link{anova.sjSDM}} function.
#' @param env Predictor variable. If \code{NULL}, assembly processes are plotted against environment, spatial uniqueness, and richness.
#' @param trait Trait variable. Plotted against species R-squared for the three processes.
#' @param Rsquared Which R-squared should be used: "McFadden" (default) or "Nagelkerke".
#' @param fractions how to handle shared fractions
#' @param cols Colors for the three assembly processes.
#' 
#' Correlation and plots of the three assembly processes (environment, space, and codist) against environmental and spatial uniqueness and richness. The importance of the three assembly processes is measured by the partial R-squared (shown in the internal structure plots).
#' Importances are available for species and sites. Custom environmental predictors or traits can be specified. Environmental predictors are plotted against site R-squared and traits are plotted against species R-squared.
#' Regression lines are estimated by 50\% quantile regression models.
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
#' @export
plotAssemblyEffects = function(object, 
                               env = NULL, 
                               trait = NULL, 
                               cols = c("#A38310", "#B42398", "#20A382")) {
  
  oldpar = par(no.readonly = TRUE)
  on.exit(par(oldpar))

  lwd = 2
  
  X = object$anova$object$settings$env$X
  XYcoords = object$anova$object$settings$spatial$X
  Y = object$anova$object$data$Y
  rr = object
  
  out = list()
  
  if(is.null(env) & is.null(trait)) {
    
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
                   ylim = c(0, 0.6), xlab = "Env uniqueness",main = "", ylab = "R2", las =1)
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
    
    graphics::plot(NULL, NULL, xlim = c(min(spatial_eigen_centered), max(spatial_eigen_centered)), ylim = c(0, 0.6), xlab = "Spatial uniqueness",main = "", ylab = "R2", las =1)
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
    
    
    graphics::plot(NULL, NULL, xlim = c(min(richness_centered), max(richness_centered)), ylim = c(0, 0.6), xlab = "Richness",main = "", ylab = "R2", las =1)
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
    if(!is.null(env)) {
      if(is.factor(env) || is.character(env)) {
        group = env
        df = data.frame(R2 = c(rr$internals$Sites$env, rr$internals$Sites$spa, rr$internals$Sites$codist),
                        part = rep(c("env", "spa", "codist"), each = nrow(rr$internals$Sites)),
                        group = rep(group, 3)
        )
        b = graphics::boxplot(R2~part+group, data =df,las = 2, col = alpha(cols[c(3, 1, 2)],0.7), xlab = "", main = "", notch = TRUE )
        b = beeswarm::beeswarm(R2~part+group, data =df,las = 2, col = cols[c(3, 1, 2)], xlab = "", main = "" , add=TRUE, method = "center", spacing = 0.3)
        graphics::legend("topright", legend = c("env", "spa", "codist"), col = cols, pch = 15, bty = "n")
      } else {
        graphics::par(mfrow = c(1, 1), mar = c(4, 4, 4, 1), xaxt= "s")
        pred = scale(env, center = TRUE, scale = FALSE)
        
        graphics::plot(NULL, NULL, xlim = c(min(pred), max(pred)), ylim = c(0, 0.6), xlab = "Predictor",main = "", ylab = "R2", las =1)
        for(i in 1:3) {
          graphics::points(pred, rr$internals$Sites[,i], col = ggplot2::alpha(cols[i], 0.2), pch = 16)
          g = qgam::qgam( Y ~ pred, data = data.frame(Y = rr$internals$Sites[,i], pred = pred), qu = 0.5, control = list(progress="none"))
          graphics::abline(a = coef(g)[c(1, 2)], col = cols[i], lwd = lwd, lty =  1*(summary(g)$p.table[2,4] > 0.05)+1 )
          out$pred[[colnames(rr$internals$Sites)[i]]] = g 
        }
        graphics::legend("topright", legend = c("env", "spa", "codist"), col = cols, pch = 15, bty = "n")
        graphics::legend("topleft", legend = c("significant", "non-significant"), col = c("black", "black"),  bty = "n", lty = c(1,2))
      }
    }
    
    if(!is.null(trait)) {
      if(is.factor(trait) || is.character(trait)) {
        group = trait
        df = data.frame(R2 = c(rr$internals$Species$env, rr$internals$Species$spa, rr$internals$Species$codist),
                        part = rep(c("env", "spa", "codist"), each = nrow(rr$internals$Species)),
                        group = rep(group, 3)
        )
        b = graphics::boxplot(R2~part+group, data =df,las = 2, col = alpha(cols[c(3, 1, 2)],0.7), xlab = "", main = "", notch = TRUE )
        b = beeswarm::beeswarm(R2~part+group, data =df,las = 2, col = cols[c(3, 1, 2)], xlab = "", main = "" , add=TRUE, method = "center", spacing = 0.3)
        graphics::legend("topright", legend = c("env", "spa", "codist"), col = cols, pch = 15, bty = "n")
      } else {
        graphics::par(mfrow = c(1, 1), mar = c(4, 4, 4, 1), xaxt= "s")
        pred = scale(trait, center = TRUE, scale = FALSE)
        
        graphics::plot(NULL, NULL, xlim = c(min(pred), max(pred)), ylim = c(0, 0.6), xlab = "Predictor",main = "", ylab = "R2", las =1)
        for(i in 1:3) {
          graphics::points(pred, rr$internals$Species[,i], col = ggplot2::alpha(cols[i], 0.2), pch = 16)
          g = qgam::qgam( Y ~ pred, data = data.frame(Y = rr$internals$Species[,i], pred = pred), qu = 0.5, control = list(progress="none"))
          graphics::abline(a = coef(g)[c(1, 2)], col = cols[i], lwd = lwd, lty =  1*(summary(g)$p.table[2,4] > 0.05)+1 )
          out$pred[[colnames(rr$internals$Species)[i]]] = g 
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

