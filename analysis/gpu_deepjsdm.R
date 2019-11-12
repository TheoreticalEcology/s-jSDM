if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(deepJSDM)
load("data_sets.RData")


result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 10L)
auc = vector("list", nrow(setup))


useGPU(1L)
set.seed(42)


counter = 1
for(i in 1:nrow(setup)) {
  sub_auc = vector("list", 10L)
  for(j in 1:10){
    .torch$cuda$empty_cache()
    X = data_sets[[counter]]$env_weights
    Y = data_sets[[counter]]$response
    tmp = data_sets[[counter]]$setup
    ### split into train and test ###
    train_X = data_sets[[counter]]$train_X
    train_Y = data_sets[[counter]]$train_Y
    test_X = data_sets[[counter]]$test_X
    test_Y = data_sets[[counter]]$test_Y
    sim = data_sets[[counter]]$sim
    counter = counter + 1L
    


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

  gpu_dmvp = list(
    setup = setup[i,],
    result_corr_acc = result_corr_acc,
    result_env = result_env,
    result_rmse_env = result_rmse_env,
    result_time= result_time,
    auc = auc
  )
  saveRDS(gpu_dmvp, "results/gpu_dmvp.RDS")
}


