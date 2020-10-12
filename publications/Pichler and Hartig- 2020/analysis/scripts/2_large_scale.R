library(sjSDM)
set.seed(42)
torch$manual_seed(42L)
torch$cuda$manual_seed(42L)


sites = c(5000, 15000,30000)
species = c(300, 500, 1000)
env = 5L
setup = expand.grid(sites, species)
colnames(setup) = c("sites", "species")
setup = setup[order(setup$sites),]

result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 10L)

for(i in 1:nrow(setup)){
  for(j in 1:10) {
    tmp = setup[i,]
    sim = simulate_SDM(env, sites = tmp$sites, species = tmp$species)
    X = sim$env_weights
    Y = sim$response
    
    model = sjSDM(Y, env =linear(X, ~X1+X2+X3+X4+X5), learning_rate = 0.003, iter = 120L, step_size = 50L,parallel = 0L,
                  device = 1L, link = "logit", sampling = 100L)
    time = model$time
    result_corr_acc[i,j] =  sim$corr_acc(getCov(model))
    ce = t(coef(model)[[1]])
    result_env[i,j] = mean(as.vector(ce[-1,] > 0) == as.vector(sim$species_weights > 0))
    result_rmse_env[i,j] =  sqrt(mean((as.vector(ce) - as.vector(rbind(matrix(0, 1L, ncol(sim$species_weights)), sim$species_weights ) ))^2))
    result_time[i,j] = time
    rm(model)
    torch$cuda$empty_cache()
  }
  result = list(
    setup = setup,
    result_corr_acc = result_corr_acc,
    result_env = result_env,
    result_rmse_env = result_rmse_env,
    result_time= result_time
  )
  saveRDS(result, "results/large_scale_logit.RDS")
}
