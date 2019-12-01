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

lrs = seq(0.0001, 0.2, length.out = 50)
result = vector("list", 50)
for(i in 1:50) {
  model = createModel(env_scaled, as.matrix(occ_high))
  model = layer_dense(model, ncol(occ_high), FALSE, FALSE, l1 = lrs[i])
  model = compileModel(model, 90L, lr = 0.01, optimizer = "adamax", l1 = lrs[i])
  model = deepJ(model, epochs = 100L, batch_size = 3L, sampling = 200L)
  weights = list(beta = model$raw_weights[[1]][[1]][[1]], sigma = model$sigma())
  rm(model)
  .torch$cuda$empty_cache()
  result[[i]] = weights
}
result[sapply(result, is.null)] = NULL
plot(NULL, NULL, xlim = c(1, 50), ylim = c(-0.1,0.1))
lines(y = smooth.spline(sapply(result, function(r) r$sigma[4,1]))$y, x = 1:length(result))
lines(y = smooth.spline(sapply(result, function(r) r$sigma[4,2]))$y, x = 1:length(result))
lines(y = smooth.spline(sapply(result, function(r) r$sigma[4,3]))$y, x = 1:length(result))
lines(y = smooth.spline(sapply(result, function(r) r$sigma[4,4]))$y, x = 1:length(result))
lines(y = smooth.spline(sapply(result, function(r) r$sigma[4,5]))$y, x = 1:length(result))
lines(y = smooth.spline(sapply(result, function(r) r$sigma[4,6]))$y, x = 1:length(result))
lines(y = smooth.spline(sapply(result, function(r) r$sigma[4,7]))$y, x = 1:length(result))


par(mfrow = c(2,2))
hh = heatmap(cov2cor(result[[30]]$sigma), keep.dendro = TRUE)

cols = viridis::viridis(20)
image(cov2cor(result[[1]]$sigma)[rev(hh$rowInd), hh$colInd],col = cols)
image(cov2cor(result[[10]]$sigma)[rev(hh$rowInd), hh$colInd], col = cols)
image(cov2cor(result[[25]]$sigma)[rev(hh$rowInd), hh$colInd], col = cols)
image(cov2cor(result[[50]]$sigma)[rev(hh$rowInd), hh$colInd], col = cols)

image((result[[50]]$beta))

