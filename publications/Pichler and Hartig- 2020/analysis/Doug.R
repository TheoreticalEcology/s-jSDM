h1 = heatmap(cov2cor(result[[40]]$sigma), keep.dendro = TRUE)
hbeta = heatmap(result[[40]]$beta, keep.dendro = TRUE)

hh = list(rowInd = h1$rowInd, colInd = h1$colInd)

# 10, 25, 40

rem = function(m) {
  m[upper.tri(m)] = NA
  m = m[nrow(m):1,]
  return(m)
}

result = readRDS("results/eDNA.RDS")
env_raw = read.csv("data/eDNA/site.info.csv")
occ = read.csv("data/eDNA/spp.present.noS.csv")

env = 
  env_raw %>% 
  select("altitude", "stump", "die_wood", "Canopy", "height", "DBH", "infestation_rate")

summary(env)

env_scaled = as.matrix(mlr::normalizeFeatures(env))


rownames(occ) = occ[,1]
occ = occ[,-1]
rates = apply(occ, 2, function(o) mean(o > 0))

# two occurences in 31 sites...
min(rates)*nrow(env_scaled)
dim(occ[,rates == min(rates)])

# First analysis with > 2 occs
occ_high = occ[,rates == min(rates)]

lrs = seq(-10, 0, length.out = 50)
f = function(x) 2^x
lrs = f(lrs)




#pdf("./figures/Fig_5.pdf", width = 7.0, height = 6.0)

layout(matrix(c(1,2,3,4, 5, 6, 7, 8), 2,4 ,byrow = TRUE), c(1,1,1,1), c(1,1))
par(mar = c(1,1,2,1), oma = c(1,1,0.3,1))

## A
cols = viridis::viridis(20)
breaks = seq(-1,1, length.out = 21)
arrow_label = function() {
  arrows(x0 = c(0.3, 0.7), x1 = c(0.0, 1.0), y0 = rep(-0.03,2), y1 = rep(-0.03,2), xpd = NA,code = 2, length =0.04)
  arrows(y0 = c(0.3, 0.7), y1 = c(0.0, 1.0), x0 = rep(-0.03,2), x1 = rep(-0.03,2), xpd = NA,code = 2, length =0.04)
  text(x = 0.3, pos = 4, y = -0.03, labels = "Species", xpd = NA, cex = 0.9)
  text(y = 0.4, pos = 3, x = -0.03, labels = "Species", xpd = NA,srt = 90, cex = 0.9)
}
leg = function(range = c(-1,1)) {
  len = length(cols)
  b = 0.4/len
  xx = matrix(c(seq(0.5, by = b, length.out = len), seq(0.5, by = b, length.out = len)+b), ncol = 2)
  for(i in 1:len) rect(xleft = xx[i,1], xright = xx[i,2], ybottom = 0.7, ytop = 0.73, col = cols[i], border = NA)
  text(x = 0.5, y = 0.7, pos = 1, labels = range[1], cex = 0.9)
  text(x = 0.9, y = 0.7, pos = 1, labels = range[2], cex = 0.9)
}




image(rem(cov2cor(result[[10]]$sigma)[hh$rowInd, hh$colInd]),col = cols, axes = FALSE, breaks = breaks)
arrow_label()
leg()
text(x = - 0.09* 1, y = 1.09 * 1, labels = "A", font = 2, xpd = NA)
text(x = 0.5, y = 1.0, pos = 3, labels = c("Penalty: 0.003"), font = 2, xpd = NA)

image(rem(cov2cor(result[[25]]$sigma)[(hh$rowInd), hh$colInd]), col = cols, axes = FALSE, breaks = breaks)
arrow_label()
leg()
text(x = 0.5, y = 1.0, pos = 3, labels = c("Penalty: 0.03"), font = 2, xpd = NA)

image(rem(cov2cor(result[[40]]$sigma)[(hh$rowInd), hh$colInd]), col = cols, axes = FALSE, breaks = breaks)
arrow_label()
leg()
text(x = 0.5, y = 1.0, pos = 3, labels = c("Penalty: 0.24"), font = 2, xpd = NA)

overall_cov = matrix(0, ncol(result[[1]]$sigma), ncol(result[[1]]$sigma))
for(i in 10:40) {
  overall_cov = overall_cov +result[[i]]$sigma
}
image(rem(overall_cov[(hh$rowInd), hh$colInd]), col = cols, axes = FALSE)
arrow_label()
leg(range = c(round(min(overall_cov[(hh$rowInd), hh$colInd]), 1), round(max(overall_cov[(hh$rowInd), hh$colInd]),1)))
text(x = 0.5, y = 1.0, pos = 3, labels = c("Importance"), font = 2, xpd = NA)


## B
#hbeta = list(rowInd = hbeta1$order, colInd = hbeta2$order)
nn = names(env)[hbeta$rowInd]
cols = viridis::viridis(20)

arrow_label = function() {
  arrows(y0 = c(0.3, 0.7), y1 = c(0.0, 1.0), x0 = rep(-0.12,2), x1 = rep(-0.12,2), xpd = NA,code = 2, length =0.04)
  text(y = 0.4, pos = 3, x = -0.12, labels = "Species", xpd = NA,srt = 90, cex = 0.9)
}
env_leg = function(){
  len = length(nn)
  text(x = seq(0,1+1/len, length.out = 8)[1:7]+0.01, y = 1.01, pos = 3, srt = 45, xpd = NA, labels = paste0("env",hbeta$rowInd), cex = 0.8)
}
leg = function(range = c(-1,1)) {
  len = length(cols)
  b = 0.4/len
  xx = matrix(c(seq(0.3, by = b, length.out = len), seq(0.3, by = b, length.out = len)+b), ncol = 2)
  for(i in 1:len) rect(xleft = xx[i,1], xright = xx[i,2], ybottom = -0.01, ytop = -0.04, col = cols[i], border = NA, xpd = NA)
  text(x = 0.3, y = -0.04, pos = 1, labels = range[1], cex = 0.9, xpd = NA)
  text(x = 0.7, y = -0.04, pos = 1, labels = range[2], cex = 0.9, xpd = NA)
}
image(result[[10]]$beta[hbeta$rowInd, hbeta$colInd], col = cols, breaks = breaks, axes = FALSE)
arrow_label()
env_leg()
text(x = - 0.18* 1 , y = 1.09 * 1, labels = "B", font = 2, xpd = NA)

leg(range = c(round(min(result[[10]]$beta[hbeta$rowInd, hbeta$colInd]), 1),round(max(result[[10]]$beta[hbeta$rowInd, hbeta$colInd]),1)))


image(result[[25]]$beta[hbeta$rowInd,hbeta$colInd], col = cols, breaks = breaks, axes = FALSE)
arrow_label()
env_leg()
leg(range = c(round(min(result[[25]]$beta[hbeta$rowInd, hbeta$colInd]), 1),round(max(result[[25]]$beta[hbeta$rowInd, hbeta$colInd]),1)))
image(result[[40]]$beta[hbeta$rowInd, hbeta$colInd], col = cols, breaks = breaks, axes = FALSE)
arrow_label()
env_leg()
leg(range = c(round(min(result[[40]]$beta[hbeta$rowInd, hbeta$colInd]), 1),round(max(result[[40]]$beta[hbeta$rowInd, hbeta$colInd]),1)))
beta = matrix(0, nrow = nrow(result[[1]]$beta), ncol = ncol(result[[1]]$beta))
for(i in 10:40) {
  beta = beta + result[[i]]$beta
}
image(beta[hbeta$rowInd, hbeta$colInd], col = cols, axes = FALSE)
arrow_label()
env_leg()
leg(range = c(round(min(beta[hbeta$rowInd, hbeta$colInd]), 1),round(max(beta[hbeta$rowInd, hbeta$colInd]),1)))

# ## C
# cols = RColorBrewer::brewer.pal(7, "Dark2")
# lwd = 3
# par(mgp = c(3,0.8,0))
# plot(NULL, NULL, xlim = c(1, 186), ylim = c(-8, 15), xaxt = "n", xaxs = "i", yaxt = "n")
# text(x = - 0.01* 186, y = 1.12 * 15, labels = "C", font = 2, xpd = NA)
# 
# lines(smooth.spline(beta[hbeta$rowInd, hbeta$colInd][1,], spar = 0.7), col = cols[1], lwd = lwd)
# lines(smooth.spline(beta[hbeta$rowInd, hbeta$colInd][2,], spar = 0.7), col = cols[2], lwd = lwd)
# lines(smooth.spline(beta[hbeta$rowInd, hbeta$colInd][3,], spar = 0.7), col = cols[3], lwd = lwd)
# lines(smooth.spline(beta[hbeta$rowInd, hbeta$colInd][4,], spar = 0.7), col = cols[4], lwd = lwd)
# lines(smooth.spline(beta[hbeta$rowInd, hbeta$colInd][5,], spar = 0.7), col = cols[5], lwd = lwd)
# lines(smooth.spline(beta[hbeta$rowInd, hbeta$colInd][6,], spar = 0.7), col = cols[6], lwd = lwd)
# lines(smooth.spline(beta[hbeta$rowInd, hbeta$colInd][7,], spar = 0.7), col = cols[7], lwd = lwd)
# legend("top", legend = paste0("env", hbeta$rowInd), col = cols, horiz = TRUE, lty = 1, border = FALSE, cex = 0.9, bty = "n")
# axis(1, labels = FALSE, at = 1:186)
# axis(2, las = 2, cex = 0.9)
# arrows(x0 = c(0.3*186, 0.7*186), x1 = c(0.1*186, 0.9*186), y0 = rep(-11,2), y1 = rep(-11,2), xpd = NA,code = 2, length =0.04)
# text(x = 0.45*186, pos = 4, y = -11, labels = "Species", xpd = NA, cex = 0.9)
# dev.off()
















# FIgure 2:

deg2rad <- function(deg) {(deg * pi) / (180)}
rad2deg <- function(rad) {(rad * 180) / (pi)}
ff = function(x){(x-min(x))/(max(x)-min(x))}


add_curve = function(p1 = coords[1,], p2 = coords[3,], n = 10, spar = 0.7, col = "black", species = TRUE, lineSeq = 5.0, lwd = 1.0) {
  xxs1 = cos(deg2rad(p1[3]))* seq(0, lineSeq, length.out = n)
  xxs2 = cos(deg2rad(p2[3]))* seq(0, lineSeq, length.out = n)
  yys1 = sin(deg2rad(p1[3]))* seq(0, lineSeq, length.out = n)
  yys2 = sin(deg2rad(p2[3]))* seq(0, lineSeq, length.out = n)
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
  sp = smooth.spline(tt[,1], tt[,2],spar = spar,df = 6, w = c(10.0, rep(0.1,nrow(tt)-2), 10.0))
  tt2 = cbind(sp$x, sp$y)
  b = tt2 %*% rot2
  lines(b[,1], b[,2], col = col, lwd = lwd)
  
  x1 = c(cos(deg2rad(p1[3]))*(lineSeq+0.1), cos(deg2rad(p1[3]))*(lineSeq+0.3))
  x2 = c(cos(deg2rad(p2[3]))*(lineSeq+0.1), cos(deg2rad(p2[3]))*(lineSeq+0.3))
  y1 = c(sin(deg2rad(p1[3]))* (lineSeq+0.1), sin(deg2rad(p1[3]))* (lineSeq+0.3))
  y2 = c(sin(deg2rad(p2[3]))* (lineSeq+0.1), sin(deg2rad(p2[3]))* (lineSeq+0.3))
  if(species){
    segments(x0 = x1[1], x1 = x1[2], y0 = y1[1], y1 = y1[2], col = "darkgrey")
    segments(x0 = x2[1], x1 = x2[2], y0 = y2[1], y1 = y2[2],  col = "darkgrey")
  }
}
add_legend = function(cols = RColorBrewer::brewer.pal(11,"Spectral"), range = c(-1,1), lineSeq = 5.0, angles = c(110, 70)){
  angles = seq(angles[1], angles[2], length.out = length(cols)+1)
  for(i in 2:length(angles)){
    xx1 = (lineSeq+0.4)*cos( seq(deg2rad(angles[i-1]),deg2rad(angles[i]) ,length.out=50) )
    xx2 = (lineSeq+0.7)*cos( seq(deg2rad(angles[i-1]),deg2rad(angles[i]) ,length.out=50) )
    yy1 = (lineSeq+0.4)*sin( seq(deg2rad(angles[i-1]),deg2rad(angles[i]) ,length.out=50)  )
    yy2 = (lineSeq+0.7)*sin( seq(deg2rad(angles[i-1]),deg2rad(angles[i]) ,length.out=50)  )
    polygon(c(xx1, rev(xx2)), c(yy1, rev(yy2)),border = NA, col = cols[i-1], xpd = NA)
    if(i == 2 || i == length(angles)) {
      if(i ==2) label = range[1]
      else label = range[2]
      tmp_a = (angles[i-1]+angles[i])/2
      text(srt = tmp_a-90, 
           x = (lineSeq+0.99)*cos(deg2rad(tmp_a)), 
           y =  (lineSeq+0.99)*sin(deg2rad(tmp_a)), 
           xpd = NA, labels = label)
    }
  }
}








#################### Start #####################


pdf(file = "figures/Doug_circles.pdf", width = 10.88, height = 8)

layout(matrix(c(1,2,3,rep(4,9)), 3,3 ,byrow = F), c(0.9,1.3,1.3), c(1,1,1))


par( mar = c(1,2,2.1,2)+0.3)

number = 30
re_scale = function(m) cov2cor(m)


#10,25,40

#A
sigma = re_scale(result[[10]]$sigma)[hh$rowInd, hh$rowInd]
sigmas = sigma[upper.tri(sigma)]
upper = order(sigmas, decreasing = TRUE)[1:number]
lower = order(sigmas, decreasing = FALSE)[1:number]
cuts = cut(sigmas, breaks = seq(-1,1,length.out = 12))
to_plot = (1:length(sigmas) %in% upper) | (1:length(sigmas) %in% lower)
levels(cuts) = viridis::viridis(11)
cuts = as.character(cuts)
n = ncol(result[[15]]$sigma)
lineSeq = 4.7
nseg = 100
plot(NULL, NULL, xlim = c(-5,5), ylim =c(-5,5),pty="s", axes = F, xlab = "", ylab = "")
text(x = 0, y = 5.7, pos = 3, xpd = NA, labels = "Penalty: 0.003")
text(x = -6, y = 5.7, pos = 3, xpd = NA, labels = "A", font = 2, cex = 1.5)

xx = lineSeq*cos( seq(0,2*pi, length.out=nseg) )
yy = lineSeq*sin( seq(0,2*pi, length.out=nseg) )
polygon(xx,yy, col= "white", border = "black", lty = 1, lwd = 1)
angles = seq(0,360,length.out = n+1)[1:(n)]
xx = cos(deg2rad(angles))*lineSeq
yy = sin(deg2rad(angles))*lineSeq

counter = 1
coords = cbind(xx, yy, angles)
for(i in 1:n) {
  for(j in i:n){
    if(i!=j) {
      if(to_plot[counter]) add_curve(coords[i,], coords[j,], col = cuts[counter], n = 5, lineSeq = lineSeq)
      counter = counter + 1
      #cat(counter, "\n")
    }
  }
}

# lineSeq = 5.0
# occ_logs = log(sort(apply(occ, 2, sum)))
# cuts = cut(occ_logs, breaks = 10)
# cols = viridis::magma(10) #colfunc(5)
# levels(cuts) = cols
# for(i in 1:length(occ_logs)){
#   p1 = coords[i,]
#   x1 = c(cos(deg2rad(p1[3]))*(lineSeq+0.1), cos(deg2rad(p1[3]))*(lineSeq+0.3))
#   y1 = c(sin(deg2rad(p1[3]))* (lineSeq+0.1), sin(deg2rad(p1[3]))* (lineSeq+0.3))
#   segments(x0 = x1[1], x1 = x1[2], y0 = y1[1], y1 = y1[2], col = as.character(cuts[i]))
# }
add_legend(viridis::viridis(11), angles = c(140,110))
text(cos(deg2rad(123))*(lineSeq+0.9), sin(deg2rad(123))*(lineSeq+0.9), labels = "covariance", pos = 2, xpd = NA)
#add_legend(cols = cols, range = c(2, 112), angles = c(70,40))
#text(cos(deg2rad(53))*(lineSeq+0.7), sin(deg2rad(55))*(lineSeq+0.7), labels = "Sp. abundance", pos = 4, xpd = NA)



#B
sigma = re_scale(result[[25]]$sigma)[hh$rowInd, hh$colInd]
sigmas = sigma[upper.tri(sigma)]
upper = order(sigmas, decreasing = TRUE)[1:number]
lower = order(sigmas, decreasing = FALSE)[1:number]
cuts = cut(sigmas, breaks = seq(-1,1,length.out = 12))
to_plot = 1:length(sigmas) %in% upper | 1:length(sigmas) %in% lower
levels(cuts) = viridis::viridis(11)
cuts = as.character(cuts)
n = ncol(result[[15]]$sigma)
lineSeq = 4.7
nseg = 100
plot(NULL, NULL, xlim = c(-5,5), ylim =c(-5,5),pty="s", axes = F, xlab = "", ylab = "")
text(x = 0, y = 5.7, pos = 3, xpd = NA, labels = "Penalty: 0.03")

text(x = -6, y = 5.7, pos = 3, xpd = NA, labels = "B", font = 2, cex = 1.5)


xx = lineSeq*cos( seq(0,2*pi, length.out=nseg) )
yy = lineSeq*sin( seq(0,2*pi, length.out=nseg) )
polygon(xx,yy, col= "white", border = "black", lty = 1, lwd = 1)
angles = seq(0,360,length.out = n+1)[1:(n)]
xx = cos(deg2rad(angles))*lineSeq
yy = sin(deg2rad(angles))*lineSeq

counter = 1
coords = cbind(xx, yy, angles)
for(i in 1:n) {
  for(j in i:n){
    if(i!=j) {
      if(to_plot[counter]) add_curve(coords[i,], coords[j,], col = cuts[counter], n = 5, lineSeq = lineSeq)
      counter = counter + 1
      #cat(counter, "\n")
    }
  }
}

add_legend(viridis::viridis(11), angles = c(140,110))
text(cos(deg2rad(123))*(lineSeq+0.9), sin(deg2rad(123))*(lineSeq+0.9), labels = "covariance", pos = 2, xpd = NA)


#C
sigma = re_scale(result[[40]]$sigma)[hh$rowInd, hh$colInd]
sigmas = sigma[upper.tri(sigma)]
upper = order(sigmas, decreasing = TRUE)[1:number]
lower = order(sigmas, decreasing = FALSE)[1:number]
cuts = cut(sigmas, breaks = seq(-1,1,length.out = 12))
to_plot = 1:length(sigmas) %in% upper | 1:length(sigmas) %in% lower
levels(cuts) = viridis::viridis(11)
cuts = as.character(cuts)
n = ncol(result[[15]]$sigma)
lineSeq = 4.7
nseg = 100
plot(NULL, NULL, xlim = c(-5,5), ylim =c(-5,5),pty="s", axes = F, xlab = "", ylab = "")
text(x = 0, y = 5.7, pos = 3, xpd = NA, labels = "Penalty: 0.24")
text(x = -6, y = 5.7, pos = 3, xpd = NA, labels = "C", font = 2, cex = 1.5)

xx = lineSeq*cos( seq(0,2*pi, length.out=nseg) )
yy = lineSeq*sin( seq(0,2*pi, length.out=nseg) )
polygon(xx,yy, col= "white", border = "black", lty = 1, lwd = 1)
angles = seq(0,360,length.out = n+1)[1:(n)]
xx = cos(deg2rad(angles))*lineSeq
yy = sin(deg2rad(angles))*lineSeq

counter = 1
coords = cbind(xx, yy, angles)
for(i in 1:n) {
  for(j in i:n){
    if(i!=j) {
      if(to_plot[counter]) add_curve(coords[i,], coords[j,], col = cuts[counter], n = 5, lineSeq = lineSeq)
      counter = counter + 1
      #cat(counter, "\n")
    }
  }
}

add_legend(viridis::viridis(11), angles = c(140,110))
text(cos(deg2rad(123))*(lineSeq+0.9), sin(deg2rad(123))*(lineSeq+0.9), labels = "covariance", pos = 2, xpd = NA)




# D
effects = matrix(NA, 50, 7)
lr_step = 25
for(i in 1:50){
  effects[i,] = apply(result[[i]]$beta, 1, function(o) sum(abs(o)))
}
turn_over = NULL
for(i in 1:49) {
  turn_over[i] = cor(effects[i,], effects[i+1,])
}

max_effects = matrix(NA, 50, n)
for(i in 1:50) max_effects[i,]= apply(result[[lr_step]]$beta,2, function(e) which.max(abs(e)))

turn_over = NULL
for(i in 1:49) {
  turn_over[i] = mean(max_effects[i,] == max_effects[i+1,])
}

effect_comb = cbind(max_effects[lr_step,], sapply(1:n, function(i) result[[lr_step]]$beta[max_effects[lr_step,i],i] ))
effect_comb_ind = order(effect_comb[,1], effect_comb[,2])
effect_comb = effect_comb[effect_comb_ind, ]
#head(effect_comb[effect_comb_ind,])

sigma = re_scale(result[[lr_step]]$sigma)[effect_comb_ind, effect_comb_ind]
sigmas = sigma[upper.tri(sigma)]
upper = order(sigmas, decreasing = TRUE)[1:number]
lower = order(sigmas, decreasing = FALSE)[1:number]
cuts = cut(sigmas, breaks = seq(-1,1,length.out = 12))
to_plot = 1:length(sigmas) %in% upper | 1:length(sigmas) %in% lower
levels(cuts) = viridis::viridis(11)
cuts = as.character(cuts)
n = ncol(result[[15]]$sigma)
lineSeq = 3.5
nseg = 100
plot(NULL, NULL, xlim = c(-5,5), ylim =c(-5,5),pty="s", axes = F, xlab = "", ylab = "")
text(x = -5, y = 5.3, pos = 3, xpd = NA, labels = "D", font = 2, cex = 1.5)

xx = lineSeq*cos( seq(0,2*pi, length.out=nseg) )
yy = lineSeq*sin( seq(0,2*pi, length.out=nseg) )
polygon(xx,yy, col= "white", border = "black", lty = 1, lwd = 1)
angles = seq(0,360,length.out = n+1)[1:(n)]
xx = cos(deg2rad(angles))*lineSeq
yy = sin(deg2rad(angles))*lineSeq

counter = 1
coords = cbind(xx, yy, angles)
for(i in 1:n) {
  for(j in i:n){
    if(i!=j) {
      if(to_plot[counter]) add_curve(coords[i,], coords[j,], col = cuts[counter], n = 5, species = TRUE, lineSeq = 3.5, lwd = 1.3)
      counter = counter + 1
    }
  }
}
cols = RColorBrewer::brewer.pal(7,"Dark2")

effect_comb2 = effect_comb
effect_comb2[,2] = ff(effect_comb[,2])
effect_comb2 = cbind(effect_comb2, effect_comb[,2])
for(i in sort(unique(max_effects[lr_step,]))) {
  sub = coords[max_effects[lr_step,effect_comb_ind] == i,]
  sub_eff = effect_comb2[max_effects[lr_step,effect_comb_ind] == i, ]
  from = sub[1,3]
  to = sub[nrow(sub),3]
  
  
  x = c((3.6+1.5*(sub_eff[,2]))*cos(deg2rad(sub[,3]) ), 
        rev((3.6+1.5/2)*cos(deg2rad(sub[,3]))))
  
  y = c((3.6+1.5*(sub_eff[,2]))*sin(deg2rad(sub[,3])),
        rev((3.6+1.5/2)*sin(deg2rad(sub[,3]))))
  #y
  polygon(x, y, xpd = NA,col = cols[i])
  text(srt = 0, 
       x = (3.6+1.7)*cos(deg2rad(sub[1,3]+4)), 
       y =  (3.6+1.7)*sin(deg2rad(sub[1,3]+4)), 
       xpd = NA, labels = round(min(sub_eff[,3]), 2), col = cols[i], cex = 0.9)
  
  text(srt = 0, 
       x = (3.6+1.7)*cos(deg2rad(sub[nrow(sub),3]-4)), 
       y =  (3.6+1.7)*sin(deg2rad(sub[nrow(sub),3]-4)), 
       xpd = NA, labels = round(max(sub_eff[,3]), 2), col = cols[i], cex = 0.9)
}
legend("bottomleft", legend = rev(colnames(env_scaled)), pch = 15, col = rev(cols), bty = "n")
#rec_cols = RColorBrewer::brewer.pal(11,"Spectral")
rec_cols = viridis::viridis(11)

x = seq(3,5, length.out = 12)
for(i in 1:length(rec_cols)){
  rect(xleft = x[i], xright = x[i+1], ybottom = -5, ytop = -5+diff(x)[1], col = rec_cols[i], xpd = NA, border = NA)
}
text(x[1],-5.2, labels = -1)
text(x[11],-5.2, labels = +1)


dev.off()



order = list(covariance_order = hh, env_order = hbeta)
save(env_scaled, occ_high, result, lrs, order, file = "eDNA.RData")
