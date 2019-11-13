if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(deepJSDM)
library(BayesComm)
load("data_sets.RData")

result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 10L)
auc = vector("list", nrow(setup))



set.seed(42)


counter = 1
for(i in 1:nrow(setup[setup$sites < 260, ])) {
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
    
    
    # BayesComm:
  
    time =
      system.time({
        model = BayesComm::BC(train_Y, train_X,model = "full", its = 10000)
      })
    
    cov = summary(model, "R")$statistics[,1]
    covFill = matrix(0,ncol(train_Y), ncol(train_Y))
    covFill[upper.tri(covFill)] = cov
    correlation = t(covFill)
    
    species_weights = matrix(NA, ncol(train_X), ncol(train_Y))
    n = paste0("B$sp",1:ncol(train_Y) )
    for(i in 1:ncol(train_Y)){
      smm = BayesComm:::summary.bayescomm(model, n[i])
      species_weights[,i]= smm$statistics[-1,1]
    }
    
    
    
    
    result_corr_acc[i,j] =  sim$corr_acc(correlation)
    result_env[i,j] = mean(as.vector(species_weights > 0) == as.vector(sim$species_weights > 0))
    result_rmse_env[i,j] =  sqrt(mean((as.vector(species_weights) - as.vector(sim$species_weights))^2))
    result_time[i,j] = time[3]
    
    pred = BayesComm:::predict.bayescomm(model, test_X)
    pred = apply(pred, 1:2, mean)
    sub_auc[[j]] = list(pred = pred, true = test_Y)
    rm(model)
    gc()
    .torch$cuda$empty_cache()
    #saveRDS(setup, file = "benchmark.RDS")
  }
  auc[[i]] = sub_auc
  
  hmsc = list(
    setup = setup[i,],
    result_corr_acc = result_corr_acc,
    result_env = result_env,
    result_rmse_env = result_rmse_env,
    result_time= result_time,
    auc = auc
  )
  saveRDS(hmsc, "results/BayesComm.RDS")
}
