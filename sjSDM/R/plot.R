#' Coefficients plot
#' 
#' Plotting coefficients returned by sjSDM model.
#' This function only for model fitted by linear, fitted by DNN is not yet supported.
#' 
#' @param x a model fitted by \code{\link{sjSDM}} 
#' @param ... Additional arguments to pass to \code{\link{plotsjSDMcoef}}. 
#' @seealso \code{\link{plotsjSDMcoef}}
#' @example /inst/examples/plot.sjSDM-example.R
#' 
#' @return No return value, called for side effects.
#' 
#' @import graphics
#' @author CAI Wang
#' @export
#' 
plot.sjSDM = function(x, ...) {
  plotsjSDMcoef(x, ...)
}

#' Internal coefficients plot
#' 
#' Plotting coefficients returned by sjSDM model.
#' This function only for model fitted by linear, fitted by DNN is not yet supported.
#' 
#' @import ggplot2
#' @param object a model fitted by \code{\link{sjSDM}} 
#' @param wrap_col Scales argument passed to wrap_col
#' @param group Define the taxonomic characteristics of a species, you need to provide a dataframe with column1 named “species” and column2 named “group”, default is NULL. For example, group[1,1]== "sp1", group[1,2]== "Mammal".
#' @param col Define colors for groups, default is NULL.
#' @param slist Select the species you want to plot, default is all, parameter is not supported yet.
#' @example /inst/examples/plot.sjSDM-example.R
#
#' @author CAI Wang

plotsjSDMcoef = function(object,wrap_col=NULL,group=NULL,col=NULL,slist=NULL) {
  stopifnot(
    inherits(object, "sjSDM"),
    inherits(object$settings$env, "linear")
  )
  oldpar = par(no.readonly = TRUE)
  on.exit(par(oldpar))
  
  if(is.null(object$se)) object=getSe(object)
  summary.se = summary(object)
  #create dataset for plot 
  effect = data.frame( Estimate=summary.se$coefmat[,1],Std.Err=summary.se$coefmat[,2],P=summary.se$coefmat[,4],rownames=rownames(summary.se$coefmat))
  
  coef = NULL
  rownames = NULL
  sep_df = do.call(rbind, strsplit(effect$rownames, split = " ", fixed = TRUE))
  colnames(sep_df) = c("species", "coef")
  effect = cbind(effect[,-4], sep_df)
  effect = effect[effect$coef!= "(Intercept)",]
  effect$coef = as.factor(effect$coef)
  effect$star = NA
  effect$star = stats::symnum(effect$P, corr = FALSE,
                               cutpoints = c(0, .001, .01, .05, .1, 1),
                               symbols = c("***","**","*","."," ")) 
  
  
  if(is.null(group)) group=NULL
  else if ( (colnames(data.frame(group)) != c("species","group"))[1]=="TRUE" |(colnames(data.frame(group)) != c("species","group"))[2]=="TRUE") {
    print ("group column's name should be 'species' and 'group'")
    group=NULL
  } else {
    effect = merge(effect, data.frame(group), by = "species", all.x = TRUE)
    if(anyNA(effect$group)) stop("There are no groups or with NAs")
    group = group[order(group$group, decreasing = TRUE), ]
    effect$species=factor(effect$species,levels= group$species)
  }
  
  # colors
  if(is.null(group) && is.null(col)) col = "grey"
  if(!is.null(group) && is.null(col)) {
    col = grDevices::palette.colors(length(unique(group$group))+1)[-1]
    names(col) = NULL
  }
  
  #if(!is.null(slist)) #effect = effect %>% dplyr::filter(species!=slist)
  
  maxy=max(effect$Estimate+effect$Std.Err)
  miny=min(effect$Estimate-effect$Std.Err)
  with(effect, {
    ggplot2::ggplot(effect,aes(x = species, y = Estimate, fill = group)) +
      geom_bar(position = position_dodge(0.6), stat="identity", width = 0.5)+
      scale_fill_manual(values=col)+
      guides(fill = guide_legend(reverse=F))+
      xlab("species") + 
      ylab("coefficients") + 
      labs(fill="Group") + 
      coord_flip(expand=F) + 
      geom_hline(aes(yintercept = 0),linetype="dashed",size=1) +
      theme_classic()+ facet_wrap(~coef, ncol = wrap_col)+ 
      geom_text(aes(y= miny-0.1, label =as.factor(star)), position = position_dodge(0.3), size = 2.5, fontface = "bold")+
      geom_errorbar(aes(ymax = Estimate + Std.Err, ymin = Estimate - Std.Err), width = 0.3)+ scale_y_continuous(limits = c(miny-0.3,maxy+0.1))
  })
}
