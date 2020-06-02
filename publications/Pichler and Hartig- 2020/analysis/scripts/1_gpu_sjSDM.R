########## Default parameter #########
# epochs = 50L
# lr = 0.01
# batch_size = 10% of data
# nlatent = 50% of n species

if(version$minor > 5) RNGkind(sampleee.kind="Rounding")
library(sjSDM)
load("data_sets2.RData")


result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 10L)
auc = vector("list", nrow(setup))



torch$manual_seed(42L)
torch$cuda$manual_seed(42L)
set.seed(42)


counter = 1
for(i in 1:nrow(setup)) {
  sub_auc = vector("list", 10L)
  for(j in 1:10){
    torch$cuda$empty_cache()
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
    
     # model = deepJ(model, epochs = 50L,batch_size = as.integer(nrow(train_X)*0.1),corr = FALSE)
    model = sjSDM(train_Y, env = linear(train_X, ~0+X1+X2+X3+X4+X5), learning_rate = 0.01,
                  iter = 50L, step_size = as.integer(nrow(train_X)*0.1), link="logit",
                  device = 0L)
      
    time = model$time
    result_corr_acc[i,j] =  sim$corr_acc(getCov(model))
    ce = t(coef(model)[[1]]) 
    result_env[i,j] = mean(as.vector(ce > 0) == as.vector(sim$species_weights > 0))
    result_rmse_env[i,j] =  sqrt(mean((as.vector(ce) - as.vector(sim$species_weights))^2))
    result_time[i,j] = time
    pred = apply(abind::abind(lapply(1:100, function(i) predict(model, newdata = test_X)), along = -1L), 2:3, mean)
    sub_auc[[j]] = list(pred = pred, true = test_Y)
    rm(model)
    gc()
    torch$cuda$empty_cache()
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
  saveRDS(gpu_dmvp, "results/gpu_sjSDM_logit.RDS")
}


