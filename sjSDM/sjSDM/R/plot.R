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
#' @param ... passed to [text()]
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
#' @param spar smoothing value, see [?stats::smooth.spline()]
#' @param col curve's color
#' @param species draw species line
#' @param radius radius
#' @param lwd curve lwd
add_curve = function(p1 = coords[1,], p2 = coords[3,], n = 10, spar = 0.7, col = "black", species = TRUE, radius = 5.0, lwd = 1.0) {
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
add_legend = function(cols = RColorBrewer::brewer.pal(11,"Spectral"), range = c(-1,1), radius = 5.0, angles = c(110, 70)){
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

add_species_arrows = function(radius = NULL, label = "Species", reverse = TRUE, start = 150, end = 270) {
  if(is.null(radius)) radius = .controlCircular$radius

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
#' @param model model of `class(model) == 'sjSDM'`, see \code{\link{sjSDM}}
#' @param radius circle's radius
#' @param main title
#' @param circleBreak circle break or not
#' @export
plotAssociations = function(model, radius = 5.0, main = NULL, circleBreak = FALSE){

  ##### circle #####
  lineSeq = 0.94*radius
  nseg = 100
  graphics::plot(NULL, NULL, xlim = c(-radius,radius), ylim =c(-radius,radius),pty="s", axes = F, xlab = "", ylab = "")
  if(!is.null(main)) graphics::text(x = 0, y = radius*1.14, pos = 3, xpd = NA, labels = main)
  xx = lineSeq*cos( seq(0,2*pi, length.out=nseg))
  yy = lineSeq*sin( seq(0,2*pi, length.out=nseg))

  graphics::polygon(xx,yy, col= "white", border = "black", lty = 1, lwd = 1)

  #### curves ####
  sigma = model$model$sigma_r
  n = ncol(sigma)

  #sigma = re_scale(result[[10]]$sigma)[order(apply(occ, 2, sum)), order(apply(occ, 2, sum))]
  sigma = stats::cov2cor(sigma)
  sigmas = sigma[upper.tri(sigma)]
  # upper = order(sigmas, decreasing = TRUE)[1:number]
  #lower = order(sigmas, decreasing = FALSE)[1:number]
  cuts = cut(sigmas, breaks = seq(-1,1,length.out = 12))
  #to_plot = (1:length(sigmas) %in% upper) | (1:length(sigmas) %in% lower)
  levels(cuts) = viridis::viridis(11)
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
        add_curve(coords[i,], coords[j,], col = cuts[counter], n = 5, radius = lineSeq)
        counter = counter + 1
        #cat(counter, "\n")
      }
    }
  }





  lineSeq = radius
  occ_logs = log(sort(apply(occ, 2, sum)))
  cuts = cut(occ_logs, breaks = 10)
  cols = viridis::magma(10) #colfunc(5)
  levels(cuts) = cols
  for(i in 1:length(occ_logs)){
    p1 = coords[i,]
    x1 = c(cos(deg2rad(p1[3]))*(lineSeq+0.1), cos(deg2rad(p1[3]))*(lineSeq+0.3))
    y1 = c(sin(deg2rad(p1[3]))* (lineSeq+0.1), sin(deg2rad(p1[3]))* (lineSeq+0.3))
    segments(x0 = x1[1], x1 = x1[2], y0 = y1[1], y1 = y1[2], col = as.character(cuts[i]), lend = 1)
  }
  add_legend(viridis::viridis(11), angles = c(140,110))
  text(cos(deg2rad(123))*(lineSeq+0.7), sin(deg2rad(123))*(lineSeq*1.14), labels = "covariance", pos = 2, xpd = NA)
  add_legend(cols = cols, range = c(2, 112), angles = c(70,40))
  text(cos(deg2rad(53))*(lineSeq+0.7), sin(deg2rad(55))*(lineSeq*1.14), labels = "Sp. abundance", pos = 4, xpd = NA)
  ### arrows

  if(isTRUE(circle_break)) {
    segments(x0 = cos(deg2rad(-1))*(lineSeq*0.96), x1 = cos(deg2rad(-1))*(lineSeq*1.18),
             y0 = sin(deg2rad(-1))*(lineSeq*0.96), y1 = sin(deg2rad(-1))*(lineSeq*1.18), xpd = NA)
    segments(x0 = cos(deg2rad(356))*(lineSeq*0.96), x1 = cos(deg2rad(356))*(lineSeq*1.18),
             y0 = sin(deg2rad(356))*(lineSeq*0.96), y1 = sin(deg2rad(356))*(lineSeq*1.18), xpd = NA)
  }
  .controlCircular = list()
  .controlCircular$radius = radius
  .controlCircular$n = n
  return(invisible(.controlCircular))
}
