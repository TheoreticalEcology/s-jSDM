if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(deepJSDM)
library(gllvm)
load("data_sets.RData")
TMB::openmp(n = 4L)

result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 10L)
auc = vector("list", nrow(setup))



set.seed(42)

dict = as.list(2:6)
names(dict) = species

counter = 1
for(i in 1:nrow(setup)) {
  sub_auc = vector("list", 10L)
  for(j in 1:10){
    
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
    
    try({
    time = system.time({
    model = gllvm::gllvm(y = train_Y, X = data.frame(train_X), family = binomial("probit"), num.lv = dict[[as.character(tmp$species)]])
    })
    
    result_corr_acc[i,j] =  sim$corr_acc(gllvm::getResidualCov(model)$cov)
    result_env[i,j] = mean(as.vector(t(coef(model)$Xcoef) > 0) == as.vector(sim$species_weights > 0))
    result_rmse_env[i,j] =  sqrt(mean((as.vector(t(coef(model)$Xcoef)) - as.vector(sim$species_weights))^2))
    result_time[i,j] = time[3]
    pred = predict.gllvm(model, newX = data.frame(test_X), type = "response")
    sub_auc[[j]] = list(pred = pred, true = test_Y)
    rm(model)
    gc()
    .torch$cuda$empty_cache()
    },silent = TRUE)
  }
  auc[[i]] = sub_auc
  
  gllvm = list(
    setup = setup[i,],
    result_corr_acc = result_corr_acc,
    result_env = result_env,
    result_rmse_env = result_rmse_env,
    result_time= result_time,
    auc = auc
  )
  saveRDS(gllvm, "results/gllvm.RDS")
}

