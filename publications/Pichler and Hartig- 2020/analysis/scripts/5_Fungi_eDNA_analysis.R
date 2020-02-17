library(sjSDM)
torch$cuda$manual_seed(42L)
torch$manual_seed(42L)

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
  model = sjSDM(X = env_scaled, Y = occ,
                formula = ~0 + precipitation_dmi+soil_ph + org_mat + soil_c + soil_p + ellenberg_l + ellenberg_f + ellenberg_n,
                df = as.integer(ncol(occ)/2), learning_rate = 0.001, iter = 100L, step_size = 8L, l1_cov = lrs[i], l2_cov = lrs[i], device = 1L)
  
  loss = unlist(model$model$logLik(env_scaled, occ))
  time = model$time
  weights = list(beta = coef(model), sigma = getCov(model), loss = loss)
  
  
  rm(model)
  torch$cuda$empty_cache()
  result[[i]] = weights
  times[[i]] = time
}


saveRDS(result, file = "results/fungi_eDNA0.RDS")
saveRDS(times, file = "results/fungi_eDNA_times0.RDS")



result = vector("list", 30)
times = vector("list", 30)
for(i in 1:30) {
  model = sjSDM(X = env_scaled, Y = occ,
                formula = ~0 + precipitation_dmi+soil_ph + org_mat + soil_c + soil_p + ellenberg_l + ellenberg_f + ellenberg_n,
                df = as.integer(ncol(occ)/2), learning_rate = 0.001, iter = 100L, step_size = 8L, 
                l1_cov = lrs[i], l2_cov = lrs[i], l1_coefs = lrs[i], l2_coefs = lrs[i],
                device = 1L)
  
  loss = unlist(model$model$logLik(env_scaled, occ))
  time = model$time
  weights = list(beta = coef(model), sigma = getCov(model), loss = loss)
  
  
  rm(model)
  torch$cuda$empty_cache()
  result[[i]] = weights
  times[[i]] = time
}


saveRDS(result, file = "results/fungi_eDNA1.RDS")
saveRDS(times, file = "results/fungi_eDNA_times1.RDS")





result = vector("list", 30)
times = vector("list", 30)
for(i in 1:30) {
  model = sjSDM(X = env_scaled, Y = occ,
                formula = ~0 + precipitation_dmi+soil_ph + org_mat + soil_c + soil_p + ellenberg_l + ellenberg_f + ellenberg_n,
                df = as.integer(ncol(occ)/2), learning_rate = 0.001, iter = 100L, step_size = 8L, 
                l1_cov = lrs[i], l2_cov = lrs[i], l1_coefs = 10*lrs[i], l2_coefs = 10*lrs[i],
                device = 1L)
  
  loss = unlist(model$model$logLik(env_scaled, occ))
  time = model$time
  weights = list(beta = coef(model), sigma = getCov(model), loss = loss)
  
  
  rm(model)
  torch$cuda$empty_cache()
  result[[i]] = weights
  times[[i]] = time
}


saveRDS(result, file = "results/fungi_eDNA2.RDS")
saveRDS(times, file = "results/fungi_eDNA_times2.RDS")







result = vector("list", 30)
times = vector("list", 30)
for(i in 1:30) {
  model = sjSDM(X = env_scaled, Y = occ,
                formula = ~0 + precipitation_dmi+soil_ph + org_mat + soil_c + soil_p + ellenberg_l + ellenberg_f + ellenberg_n,
                df = as.integer(ncol(occ)/2), learning_rate = 0.001, iter = 100L, step_size = 8L, 
                l1_cov = lrs[i], l2_cov = lrs[i], l1_coefs = 50*lrs[i], l2_coefs = 50*lrs[i],
                device = 1L)
  
  loss = unlist(model$model$logLik(env_scaled, occ))
  time = model$time
  weights = list(beta = coef(model), sigma = getCov(model), loss = loss)
  
  
  rm(model)
  torch$cuda$empty_cache()
  result[[i]] = weights
  times[[i]] = time
}


saveRDS(result, file = "results/fungi_eDNA3.RDS")
saveRDS(times, file = "results/fungi_eDNA_times3.RDS")





