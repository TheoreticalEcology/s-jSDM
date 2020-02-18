library(sjSDM)
set.seed(42)
torch$manual_seed(42L)
torch$cuda$manual_seed(42L)


sites = seq(5000, 31000, by = 2000)
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
    
    
    # model = deepJ(model, epochs = 50L,batch_size = as.integer(nrow(train_X)*0.1),corr = FALSE)
    model = sjSDM(X, Y, formula = ~0+X1+X2+X3+X4+X5, learning_rate = 0.01, 
                  df = as.integer(tmp$species/2),iter = 50L, step_size = 75L,parallel = 0L,
                  device = 2L)
    time = model$time
    result_corr_acc[i,j] =  sim$corr_acc(getCov(model))
    result_env[i,j] = mean(as.vector(coef(model)[[1]] > 0) == as.vector(sim$species_weights > 0))
    result_rmse_env[i,j] =  sqrt(mean((as.vector(coef(model)[[1]]) - as.vector(sim$species_weights))^2))
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
  saveRDS(result, "results/large_scale.RDS")
}
