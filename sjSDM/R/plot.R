#' Coeffect plot
#' 
#' Plotting coeffects return by sjSDM model.
#' This function only for model fitted by linear, fitted by DNN is not yet supported.
#' 
#' @import tidyr
#' @import dplyr
#' @import ggplot2
#' @importFrom magrittr `%>%`
#' @param object a model fitted by \code{\link{sjSDM}} 
#' @param ... Additional arguments to pass to \code{\link{plot.sjSDM.coef}}. 
#' @seealso \code{\link{plot.sjSDM.coef}}
#' @example /inst/examples/plot.sjSDM-emample.R
#
#' @author CAI Wang
#' @export
#' 
plot.sjSDM = function(object, ...) {
  plot.sjSDM.coef(object, ...)
}

#' Coeffect plot
#' 
#' Plotting coeffects return by sjSDM model.
#' This function only for model fitted by linear, fitted by DNN is not yet supported.
#' 
#' @import tidyr
#' @import dplyr
#' @import ggplot2
#' @importFrom magrittr `%>%`
#' @param object a model fitted by \code{\link{sjSDM}} 
#' @param wrap_col Scales argument passed to wrap_col
#' @param group Define the taxonomic characteristics of a species, you need to provide a dataframe with column1 named “species” and column2 named “group”, default is NULL. For example, group[1,1]== "sp1", group[1,2]== "Mammal".
#' @param col Define colors for groups, default is NULL.
#' @param slist Select the species you want to plot, default is all, parameter is not supported yet.
#' @example /inst/examples/plot.sjSDM-emample.R
#
#' @author CAI Wang
#' @export

plot.sjSDM.coef = function(object,wrap_col=NULL,group=NULL,col=NULL,slist=NULL) {
  stopifnot(
    inherits(object, "sjSDM"),
    inherits(object$settings$env, "linear")
  )
  
  if(is.null(object$se)) object=getSe(object)
  summary.se=summary(object)
  #create dataset for plot 
  effect = data.frame( Estimate=summary.se$coefmat[,1],Std.Err=summary.se$coefmat[,2],P=summary.se$coefmat[,4],rownames=rownames(summary.se$coefmat))
  
  effect= effect %>% tidyr::separate(col = rownames, into = c("species", "coef"), sep = " ") %>% dplyr::filter(coef != "(Intercept)") %>% dplyr::mutate(coef=as.factor(coef),star=NA)
  
  effect$star <- stats::symnum(effect$P, corr = FALSE,
                               cutpoints = c(0, .001, .01, .05, .1, 1),
                               symbols = c("***","**","*","."," ")) 
  
  
  if(is.null(group)) group=NULL
  else if ( (colnames(data.frame(group)) != c("species","group"))[1]=="TRUE" |(colnames(data.frame(group)) != c("species","group"))[2]=="TRUE") {
    print ("group column's name should be 'species' and 'group'")
    group=NULL
  }
  else {
    effect=dplyr::left_join(effect,data.frame(group),by="species")
    if(anyNA(effect$group)) stop("There are no groups or with NAs")
    group= dplyr::arrange(group,desc(group))
    effect$species=factor(effect$species,levels= group$species)
  }
  
  if(is.null(col))  
    col <- RColorBrewer::brewer.pal(10, "Paired") 
  else col=col
  
  #if(!is.null(slist)) #effect = effect %>% dplyr::filter(species!=slist)
  
  maxy=max(effect$Estimate+effect$Std.Err)
  miny=min(effect$Estimate-effect$Std.Err)
  
  ggplot2::ggplot(effect,aes(x = species, y = Estimate, fill = group)) +
    geom_bar(position = position_dodge(0.6), stat="identity", width = 0.5)+
    scale_fill_manual(values=col)+
    guides(fill = guide_legend(reverse=F))+
    xlab("species") + 
    ylab("coef") + 
    labs(fill="Group") + 
    coord_flip(expand=F) + 
    geom_hline(aes(yintercept = 0),linetype="dashed",size=1) +
    theme_classic()+ facet_wrap(~coef, ncol = wrap_col)+ 
    geom_text(aes(y= miny-0.1, label =as.factor(star)), position = position_dodge(0.3), size = 2.5, fontface = "bold")+
    geom_errorbar(aes(ymax = Estimate + Std.Err, ymin = Estimate - Std.Err), width = 0.3)+ scale_y_continuous(limits = c(miny-0.3,maxy+0.1))
}

#' deg2rad
#' degree to rad
#' @param deg degree
deg2rad = function(deg) {(deg * pi) / (180)}

#' rad2deg
#' rad to degree
#' @param rad rad
rad2deg = function(rad) {(rad * 180) / (pi)}


#ff = function(x){(x-min(x))/(max(x)-min(x))}

#' curve_text
#' plot curved text
#' @param pos position in degree
#' @param label label
#' @param radius radius
#' @param reverse in reverse order
#' @param middle text in the middle
#' @param extend extend char lengths, default 1.1
#' @param ... graphics::text
#' @export

curve_text = function(pos = 0, label = "", radius = 5.0, reverse = FALSE,middle = FALSE,extend = 1.1, ...){
  # inspired by plotrix package
  chars = strsplit(label,split = "")[[1]]
  char_lens = graphics::strwidth(chars)*extend
  char_angles = char_lens / radius
  changrang = range(char_angles)
  char_angles[char_angles < changrang[2]/2] = changrang[2]/2

  if(middle & reverse) pos = pos - rad2deg(sum(char_angles)/2)
  if(middle & !reverse) pos = pos + rad2deg(sum(char_angles)/2)

  if(reverse) {
    angles = c(deg2rad(pos), deg2rad(pos)+cumsum(char_angles)[-length(chars)])
    angles = angles + char_angles/2
  } else {
    angles = c(deg2rad(pos), deg2rad(pos)-cumsum(char_angles)[-length(chars)])
    angles = angles - char_angles/2
  }

  for(i in 1:length(chars)) graphics::text(label = chars[i],
                                           x = cos((angles[i]))*(radius),
                                           srt = rad2deg(angles[i]) - 90+ 180*reverse,
                                           y = sin((angles[i]))*(radius),
                                           xpd = NA, adj = c(0.5, 0.5),...)
  return(max(angles))

}


#' add_curve
#' curve plotting 'engine'
#' @param p1 first point
#' @param p2 second points
#' @param n number of points for spline
#' @param spar smoothing value
#' @param col curve's color
#' @param species draw species line
#' @param radius radius
#' @param lwd curve lwd
add_curve = function(p1 = NULL, p2 = NULL, n = 10, spar = 0.7, col = "black", species = TRUE, radius = 5.0, lwd = 1.0) {
  xxs1 = cos(deg2rad(p1[3]))* seq(0, radius, length.out = n)
  xxs2 = cos(deg2rad(p2[3]))* seq(0, radius, length.out = n)
  yys1 = sin(deg2rad(p1[3]))* seq(0, radius, length.out = n)
  yys2 = sin(deg2rad(p2[3]))* seq(0, radius, length.out = n)
  x = c(rev(xxs1), xxs2[-1])
  y = c(rev(yys1), yys2[-1])
  m = (p1[2] - p2[2])/(p1[1] - p2[1])
  a = rad2deg(atan(m))
  a = -(a+180)
  alpha = deg2rad(a)
  alpha2 = deg2rad(-a)
  rot = matrix(c(cos((alpha)), -sin((alpha)), sin((alpha)), cos((alpha))),2,2)
  rot2 = matrix(c(cos((alpha2)), -sin((alpha2)), sin((alpha2)), cos((alpha2))),2,2)
  tt = cbind(x,y) %*% rot
  sp = stats::smooth.spline(tt[,1], tt[,2],spar = spar,df = 6, w = c(10.0, rep(0.1,nrow(tt)-2), 10.0))
  tt2 = cbind(sp$x, sp$y)
  b = tt2 %*% rot2
  graphics::lines(b[,1], b[,2], col = col, lwd = lwd)

  x1 = c(cos(deg2rad(p1[3]))*(radius+0.1), cos(deg2rad(p1[3]))*(radius+0.3))
  x2 = c(cos(deg2rad(p2[3]))*(radius+0.1), cos(deg2rad(p2[3]))*(radius+0.3))
  y1 = c(sin(deg2rad(p1[3]))* (radius+0.1), sin(deg2rad(p1[3]))* (radius+0.3))
  y2 = c(sin(deg2rad(p2[3]))* (radius+0.1), sin(deg2rad(p2[3]))* (radius+0.3))
  if(species){
    graphics::segments(x0 = x1[1], x1 = x1[2], y0 = y1[1], y1 = y1[2], col = "darkgrey")
    graphics::segments(x0 = x2[1], x1 = x2[2], y0 = y2[1], y1 = y2[2],  col = "darkgrey")
  }
}


#' add_legend
#' add legend to circular plot
#'
#' @param cols colors for gradients
#' @param range gradient range
#' @param radius radius
#' @param angles angles, start and end values in degree
#' @export
add_legend = function(cols = 1:11, range = c(-1,1), radius = 5.0, angles = c(110, 70)){
  angles = seq(angles[1], angles[2], length.out = length(cols)+1)
  for(i in 2:length(angles)){
    xx1 = (radius+0.4)*cos( seq(deg2rad(angles[i-1]),deg2rad(angles[i]) ,length.out=50) )
    xx2 = (radius+0.7)*cos( seq(deg2rad(angles[i-1]),deg2rad(angles[i]) ,length.out=50) )
    yy1 = (radius+0.4)*sin( seq(deg2rad(angles[i-1]),deg2rad(angles[i]) ,length.out=50)  )
    yy2 = (radius+0.7)*sin( seq(deg2rad(angles[i-1]),deg2rad(angles[i]) ,length.out=50)  )
    graphics::polygon(c(xx1, rev(xx2)), c(yy1, rev(yy2)),border = NA, col = cols[i-1], xpd = NA)
    if(i == 2 || i == length(angles)) {
      if(i ==2) label = range[1]
      else label = range[2]
      tmp_a = (angles[i-1]+angles[i])/2
      graphics::text(srt = tmp_a-90,
                     x = (radius+0.99)*cos(deg2rad(tmp_a)),
                     y =  (radius+0.99)*sin(deg2rad(tmp_a)),
                     xpd = NA, labels = label)
    }
  }
}

#' add_species_arrows
#' add species arrows to circle
#' @param radius radius
#' @param label label between arrows
#' @param reverse reverse label
#' @param start start point for arrow in degree
#' @param end end point for arrow in degree
#' @export

add_species_arrows = function(radius = 5.0, label = "Species", reverse = TRUE, start = 150, end = 270) {

  # first
  angles = seq(150,195,length.out = 100)
  xx = cos(deg2rad(angles))*(radius+0.6)
  yy = sin(deg2rad(angles))*(radius+0.6)
  graphics::lines(xx, yy, xpd = NA)
  end = curve_text(195, label,radius = radius*1.12,reverse = reverse)
  # second
  angles = seq(rad2deg(end)+3,rad2deg(end)+45+8,length.out = 100)
  xx = cos(deg2rad(angles))*(radius*1.12)
  yy = sin(deg2rad(angles))*(radius*1.12)
  graphics::lines(xx, yy, xpd = NA)
  arrow_angle = max(angles)-2.8
  graphics::polygon(x = c(cos(deg2rad(arrow_angle))*(radius*1.10), cos(deg2rad(arrow_angle))*(radius*1.14), cos(deg2rad(max(angles)))*(radius*1.12), cos(deg2rad(arrow_angle))*(radius*1.10)),
                    y = c(sin(deg2rad(arrow_angle))*(radius*1.10), sin(deg2rad(arrow_angle))*(radius*1.14), sin(deg2rad(max(angles)))*(radius*1.12), sin(deg2rad(arrow_angle))*(radius*1.10)),col = "black", xpd = NA)
}


#' plotAssociations
#' plot species-species associations
#'
#' @param sigma species-species covariance matrix
#' @param radius circle's radius
#' @param main title
#' @param circleBreak circle break or not
#' @param top top negative and positive associations
#' @param occ species occurence data
#' @param cols_association col gradient for association lines
#' @param cols_occurrence col gradient for species 
#' @param lwd_occurrence lwd for occurrence lines
#' @param species_indices indices for sorting species
#' @export
plotAssociations = function(sigma, radius = 5.0, main = NULL, 
                            circleBreak = FALSE, top = 10L, occ = NULL, 
                            cols_association = c("#FF0000", "#BF003F", "#7F007F", "#3F00BF", "#0000FF"),
                            cols_occurrence = c( "#BEBEBE", "#8E8E8E", "#5F5F5F", "#2F2F2F", "#000000"),
                            lwd_occurrence = 1.0,
                            species_indices = NULL
                            ){

  ##### circle #####
  
  lineSeq = 0.94*radius
  nseg = 100
  graphics::plot(NULL, NULL, xlim = c(-radius,radius), ylim =c(-radius,radius),pty="s", axes = F, xlab = "", ylab = "")
  if(!is.null(main)) graphics::text(x = 0, y = radius*1.14, pos = 3, xpd = NA, labels = main)
  xx = lineSeq*cos( seq(0,2*pi, length.out=nseg))
  yy = lineSeq*sin( seq(0,2*pi, length.out=nseg))

  graphics::polygon(xx,yy, col= "white", border = "black", lty = 1, lwd = 1)

  #### curves ####
  n = ncol(sigma)
  
  if(!is.null(species_indices))
    sigma = sigma[species_indices, species_indices]
  else {
    ### occ ###
    if(!is.null(occ)) {
      species_indices = sort(apply(occ, 2, sum))
      sigma = sigma[species_indices, species_indices]
    }
  }
  
  #sigma = re_scale(result[[10]]$sigma)[order(apply(occ, 2, sum)), order(apply(occ, 2, sum))]
  sigma = stats::cov2cor(sigma)
  sigmas = sigma[upper.tri(sigma)]
  upper = order(sigmas, decreasing = TRUE)[1:top]
  lower = order(sigmas, decreasing = FALSE)[1:top]
  cuts = cut(sigmas, breaks = seq(-1,1,length.out = length(cols_association) + 1))
  to_plot = (1:length(sigmas) %in% upper) | (1:length(sigmas) %in% lower)
  levels(cuts) = cols_association
  cuts = as.character(cuts)

  angles = seq(0,355,length.out = n+1)[1:(n)]
  xx = cos(deg2rad(angles))*lineSeq
  yy = sin(deg2rad(angles))*lineSeq
  counter = 1
  coords = cbind(xx, yy, angles)
  for(i in 1:n) {
    for(j in i:n){
      if(i!=j) {
        #if(to_plot[counter]) add_curve(coords[i,], coords[j,], col = cuts[counter], n = 5, lineSeq = lineSeq)
        if(to_plot[counter]) add_curve(coords[i,], coords[j,], col = cuts[counter], n = 5, radius = lineSeq)
        counter = counter + 1
        #cat(counter, "\n")
      }
    }
  }
  
  
  ### occ ###
  if(!is.null(occ)) {
    lineSeq = radius
    occ_abs = sort(apply(occ, 2, sum))
    occ_logs = log(occ_abs)
    cuts = cut(occ_logs, breaks = length(cols_occurrence))
    cols = cols_occurrence #colfunc(5)
    levels(cuts) = cols_occurrence
    for(i in 1:length(occ_logs)){
      p1 = coords[i,]
      x1 = c(cos(deg2rad(p1[3]))*(lineSeq+0.1), cos(deg2rad(p1[3]))*(lineSeq+0.3))
      y1 = c(sin(deg2rad(p1[3]))* (lineSeq+0.1), sin(deg2rad(p1[3]))* (lineSeq+0.3))
      graphics::segments(x0 = x1[1], x1 = x1[2], y0 = y1[1], y1 = y1[2], col = as.character(cuts[i]), lend = 1, lwd = lwd_occurrence)
    }
    add_legend(cols_association, angles = c(140,110))
    graphics::text(cos(deg2rad(123))*(lineSeq+0.7), sin(deg2rad(123))*(lineSeq*1.14), labels = "covariance", pos = 2, xpd = NA)
    add_legend(cols = cols, range = c(min(occ_abs), max(occ_abs)), angles = c(70,40))
    graphics::text(cos(deg2rad(53))*(lineSeq+0.7), sin(deg2rad(55))*(lineSeq*1.14), labels = "Sp. abundance", pos = 4, xpd = NA)
  }
  
  ### arrows

  if(isTRUE(circleBreak)) {
    graphics::segments(x0 = cos(deg2rad(-1))*(lineSeq*0.96), x1 = cos(deg2rad(-1))*(lineSeq*1.18),
             y0 = sin(deg2rad(-1))*(lineSeq*0.96), y1 = sin(deg2rad(-1))*(lineSeq*1.18), xpd = NA)
    graphics::segments(x0 = cos(deg2rad(356))*(lineSeq*0.96), x1 = cos(deg2rad(356))*(lineSeq*1.18),
             y0 = sin(deg2rad(356))*(lineSeq*0.96), y1 = sin(deg2rad(356))*(lineSeq*1.18), xpd = NA)
  }
  controlCircular = list()
  controlCircular$radius = radius
  controlCircular$n = n
  return(invisible(controlCircular))
}

