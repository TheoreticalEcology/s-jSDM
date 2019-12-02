library(deepJSDM)
library(tidyverse)
useGPU(1L)


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

result = vector("list", 50)
for(i in 1:50) {
  model = createModel(env_scaled, as.matrix(occ_high))
  model = layer_dense(model, ncol(occ_high), FALSE, FALSE, l1 = lrs[i])
  model = compileModel(model, 90L, lr = 0.01, optimizer = "adamax", l1 = lrs[i], l2 = lrs[i])
  model = deepJ(model, epochs = 100L, batch_size = 3L, sampling = 200L)
  weights = list(beta = model$raw_weights[[1]][[1]][[1]], sigma = model$sigma())
  rm(model)
  .torch$cuda$empty_cache()
  result[[i]] = weights
}
result[sapply(result, is.null)] = NULL
plot(NULL, NULL, xlim = c(1, 50), ylim = c(-0.1,0.1))



hh = heatmap(cov2cor(result[[30]]$sigma), keep.dendro = TRUE)
hbeta = heatmap(result[[30]]$beta, keep.dendro = TRUE)


#par(mfrow = c(2,4))
pdf("figures/Fig_4.pdf", width = 7.0, height = 6.0)
rem = function(m) {
  m[upper.tri(m)] = NA
  m = m[nrow(m):1,]
  return(m)
}





















# Figure 4


pdf("figures/Fig_4.pdf", width = 7.0, height = 6.0)

layout(matrix(c(1,2,3,4, 5, 6, 7, 8, 9, 9,9,9), 3,4 ,byrow = TRUE), c(1,1,1,1), c(1,1,1))
par(mar = c(1,1,2,1), oma = c(1,1,0.3,1))

## A
cols = viridis::viridis(20)

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


image(rem(cov2cor(result[[1]]$sigma)[hh$rowInd, hh$colInd]),col = cols, axes = FALSE, breaks = breaks)
arrow_label()
leg()
text(x = - 0.09* 1, y = 1.09 * 1, labels = "A", font = 2, xpd = NA)
text(x = 0.5, y = 1.0, pos = 3, labels = c("Penalty: 9e-4"), font = 2, xpd = NA)

image(rem(cov2cor(result[[15]]$sigma)[(hh$rowInd), hh$colInd]), col = cols, axes = FALSE, breaks = breaks)
arrow_label()
leg()
text(x = 0.5, y = 1.0, pos = 3, labels = c("Penalty: 7e-3"), font = 2, xpd = NA)


image(rem(cov2cor(result[[30]]$sigma)[(hh$rowInd), hh$colInd]), col = cols, axes = FALSE, breaks = breaks)
arrow_label()
leg()
text(x = 0.5, y = 1.0, pos = 3, labels = c("Penalty: 0.059"), font = 2, xpd = NA)


overall_cov = matrix(0, ncol(occ_high), ncol(occ_high))
for(i in 10:50) {
  overall_cov = overall_cov +result[[i]]$sigma
}
image(rem(overall_cov[(hh$rowInd), hh$colInd]), col = cols, axes = FALSE)
arrow_label()
leg(range = c(round(min(overall_cov[(hh$rowInd), hh$colInd]), 1), round(max(overall_cov[(hh$rowInd), hh$colInd]),1)))
text(x = 0.5, y = 1.0, pos = 3, labels = c("Importance"), font = 2, xpd = NA)



## B
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
image(result[[1]]$beta[hbeta$rowInd, hbeta$colInd], col = cols, breaks = breaks, axes = FALSE)
arrow_label()
env_leg()
text(x = - 0.18* 1 , y = 1.09 * 1, labels = "B", font = 2, xpd = NA)

leg(range = c(round(min(result[[1]]$beta[hbeta$rowInd, hbeta$colInd]), 1),round(max(result[[1]]$beta[hbeta$rowInd, hbeta$colInd]),1)))
image(result[[25]]$beta[hbeta$rowInd,hbeta$colInd], col = cols, breaks = breaks, axes = FALSE)
arrow_label()
env_leg()
leg(range = c(round(min(result[[25]]$beta[hbeta$rowInd, hbeta$colInd]), 1),round(max(result[[1]]$beta[hbeta$rowInd, hbeta$colInd]),1)))
image(result[[30]]$beta[hbeta$rowInd, hbeta$colInd], col = cols, breaks = breaks, axes = FALSE)
arrow_label()
env_leg()
leg(range = c(round(min(result[[30]]$beta[hbeta$rowInd, hbeta$colInd]), 1),round(max(result[[1]]$beta[hbeta$rowInd, hbeta$colInd]),1)))
beta = matrix(0, nrow = ncol(env_scaled), ncol = ncol(occ_high))
for(i in 1:50) {
  beta = beta + result[[i]]$beta  
}
image(beta[hbeta$rowInd, hbeta$colInd], col = cols, axes = FALSE)
arrow_label()
env_leg()
leg(range = c(round(min(beta[hbeta$rowInd, hbeta$colInd]), 1),round(max(beta[hbeta$rowInd, hbeta$colInd]),1)))




## C
cols = RColorBrewer::brewer.pal(7, "Dark2")
lwd = 3
par(mgp = c(3,0.8,0))
plot(NULL, NULL, xlim = c(1, 186), ylim = c(-8, 15), xaxt = "n", xaxs = "i", yaxt = "n")
text(x = - 0.01* 186, y = 1.12 * 15, labels = "C", font = 2, xpd = NA)

lines(smooth.spline(beta[hbeta$rowInd, hbeta$colInd][1,], spar = 0.7), col = cols[1], lwd = lwd)
lines(smooth.spline(beta[hbeta$rowInd, hbeta$colInd][2,], spar = 0.7), col = cols[2], lwd = lwd)
lines(smooth.spline(beta[hbeta$rowInd, hbeta$colInd][3,], spar = 0.7), col = cols[3], lwd = lwd)
lines(smooth.spline(beta[hbeta$rowInd, hbeta$colInd][4,], spar = 0.7), col = cols[4], lwd = lwd)
lines(smooth.spline(beta[hbeta$rowInd, hbeta$colInd][5,], spar = 0.7), col = cols[5], lwd = lwd)
lines(smooth.spline(beta[hbeta$rowInd, hbeta$colInd][6,], spar = 0.7), col = cols[6], lwd = lwd)
lines(smooth.spline(beta[hbeta$rowInd, hbeta$colInd][7,], spar = 0.7), col = cols[7], lwd = lwd)
legend("top", legend = paste0("env", hbeta$rowInd), col = cols, horiz = TRUE, lty = 1, border = FALSE, cex = 0.9, bty = "n")
axis(1, labels = FALSE, at = 1:186)
axis(2, las = 2, cex = 0.9)
#text(y = -10, x = (1:186)-0.5, pos = 1, labels = paste0("Sp-", hbeta$rowInd), srt = 45, xpd = NA)
arrows(x0 = c(0.3*186, 0.7*186), x1 = c(0.1*186, 0.9*186), y0 = rep(-11,2), y1 = rep(-11,2), xpd = NA,code = 2, length =0.04)
text(x = 0.45*186, pos = 4, y = -11, labels = "Species", xpd = NA, cex = 0.9)

dev.off()




