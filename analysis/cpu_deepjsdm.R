if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(deepJSDM)
sites = c(50, 70, 100, 140, 180, 260, 320, 400, 500)
species = c(0.1, 0.2, 0.3, 0.4,0.5)
env = c(3,5,7)


setup = expand.grid(sites, species, env)
colnames(setup) = c("sites", "species", "env")
setup = setup[order(setup$sites,decreasing = FALSE),]
result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 10L)
auc = vector("list", nrow(setup))


useCPU()
.torch$set_num_threads(4L) 
set.seed(42)



for(i in 1:nrow(setup)) {
  sub_auc = vector("list", 10L)
  for(j in 1:10){
    .torch$cuda$empty_cache()
    tmp = setup[i,]
    sim = simulate_SDM(env = tmp$env,sites = 2*tmp$sites,species = as.integer(tmp$species*tmp$sites))
    X = sim$env_weights
    Y = sim$response
    
    ### split into train and test ###
    indices = sample.int(nrow(X), 0.5*nrow(X))
    train_X = X[indices, ]
    train_Y = Y[indices, ]
    test_X = X[-indices, ]
    test_Y = Y[-indices, ]
    
    
    model = createModel(train_X, train_Y)
    model = layer_dense(model,ncol(train_Y),FALSE, FALSE)
    model = compileModel(model, nLatent = as.integer(tmp$species*tmp$sites*0.5),lr = 1.0,optimizer = "LBFGS",reset = TRUE)
    time = system.time({
      model = deepJ(model, epochs = 15L,batch_size = nrow(train_X),corr = FALSE)
    })
    
    result_corr_acc[i,j] =  sim$corr_acc(model$sigma())
    result_env[i,j] = mean(as.vector(model$raw_weights[[1]][[1]][[1]] > 0) == as.vector(sim$species_weights > 0))
    result_rmse_env[i,j] =  sqrt(mean((as.vector(model$raw_weights[[1]][[1]][[1]]) - as.vector(sim$species_weights))^2))
    result_time[i,j] = time[3]
    pred = predict(model, test_X)
    sub_auc[[j]] = list(pred = pred, true = test_Y)
    rm(model)
    gc()
    .torch$cuda$empty_cache()
    #saveRDS(setup, file = "benchmark.RDS")
  }
  auc[[i]] = sub_auc
  
  cpu_dmvp = list(
    setup = setup,
    result_corr_acc = result_corr_acc,
    result_env = result_env,
    result_rmse_env = result_rmse_env,
    result_time= result_time,
    auc = auc
  )
  saveRDS(cpu_dmvp, "results/cpu_dmvp.RDS")
}

