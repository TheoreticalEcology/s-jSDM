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

lrs = seq(-18, -1, length.out = 30)
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
saveRDS(times, file = "results/fungi_eDNA_times.RDS")


