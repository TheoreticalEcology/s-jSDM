library(deepJSDM)
set.seed(42)
useGPU(0L)
.torch$manual_seed(seed)

sites = seq(5000, 31000, by = 2000)
species = c(300, 500, 1000)
e = 5L
setup = expand.grid(sites, species)
colnames(setup) = c("sites", "species")
setup = setup[order(setup$sites),]

result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 1L)

for(i in 1:nrow(setup)){
  tmp = setup[i,]
  sim = simulate_SDM(env, sites = tmp$sites, species = tmp$species)
  X = sim$env_weights
  Y = sim$response
  
  model = createModel(X, Y)
  model = layer_dense(model, ncol(Y),FALSE, FALSE)
  model = compileModel(model, nLatent = as.integer(tmp$species/2),lr = 0.01,optimizer = "adamax",reset = TRUE)
  time = system.time({
    model = deepJ(model, epochs = 50L,batch_size = 75L,corr = FALSE, parallel = 3L)
  })
  
  result_corr_acc[i,1] =  sim$corr_acc(model$sigma())
  result_env[i,1] = mean(as.vector(model$raw_weights[[1]][[1]][[1]] > 0) == as.vector(sim$species_weights > 0))
  result_rmse_env[i,1] =  sqrt(mean((as.vector(model$raw_weights[[1]][[1]][[1]]) - as.vector(sim$species_weights))^2))
  result_time[i,1] = time[3]
  rm(model)
  .torch$cuda$empty_cache()
}
result = list(
  setup = setup,
  result_corr_acc = result_corr_acc,
  result_env = result_env,
  result_rmse_env = result_rmse_env,
  result_time= result_time
)
saveRDS(result, "results/large_scale.RDS")