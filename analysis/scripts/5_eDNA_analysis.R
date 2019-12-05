library(deepJSDM)
library(tidyverse)
useGPU(0L)
set.seed(42L)
.torch$cuda$manual_seed(42L)
.torch$manual_seed(42L)

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

saveRDS(result, file = "results/eDNA.RDS")



# par(mfrow = c(1,1))
# plot(NULL, NULL, xlim = c(1, 186), ylim = c(0, 1), xaxt = "n", xaxs = "i", yaxt = "n", yaxs = "i")
# sb = function(m,i){
#   v = abs(m/sum(abs(m)))
#   v = c(0, v)
#   for(j in 2:8){
#     rect(xleft = i-0.5, xright = i+0.5, ybottom = sum(v[1:j-1]), ytop = sum(v[1:j]), col = cols[j-1],border = "black")
#   }
#   return(v)
# }
# sb2 = 
#   function(m,i){
#     v = abs(m/sum(abs(m)))
#     v = c(0, v)
#     rect(xleft = i-0.5, xright = i+0.5, ybottom = 0, ytop = 1L, col = cols[which.max(v)-1],border = "black")
#     return(v)
#   }
# rr = matrix(NA, 186, 7)
# for(i in 1:ncol(beta[hbeta$rowInd, hbeta$colInd])){
#   rr[i,] = sb(beta[hbeta$rowInd, hbeta$colInd][,i],i )[2:8]
# }
