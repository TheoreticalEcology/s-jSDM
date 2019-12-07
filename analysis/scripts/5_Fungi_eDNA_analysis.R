library(deepJSDM)
useGPU(2L)
.torch$cuda$manual_seed(42L)
.torch$manual_seed(42L)

load("data/eDNA/SDM_data.RData")

occ = ttab_ds[[7]]
dim(occ)
rates = apply(occ, 2, mean)
occ = occ[,rates > 1/125]
dim(occ)

env_scaled = as.matrix(mlr::normalizeFeatures(env2))

lrs = seq(-12, -1, length.out = 30)
f = function(x) 2^x
lrs = f(lrs)

result = vector("list", 30)
times = vector("list", 30)
for(i in 1:30) {
  model = createModel(env_scaled, occ)
  model = layer_dense(model, hidden = ncol(occ), FALSE, FALSE, l1 = lrs[i], l2 = lrs[i])
  model = compileModel(model, nLatent = as.integer(ncol(occ)/2), lr = 0.001, optimizer = "adamax",l1 = lrs[i], l2 = lrs[i])
  
  time = system.time({model = deepJ(model, epochs = 50L, batch_size = 8L, sampling = 100L)})
  
  weights = list(beta = model$raw_weights[[1]][[1]][[1]], sigma = model$sigma())
  rm(model)
  .torch$cuda$empty_cache()
  result[[i]] = weights
  times[[i]] = time
}

saveRDS(result, file = "results/fungi_eDNA.RDS")



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
