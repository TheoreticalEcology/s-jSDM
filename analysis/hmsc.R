if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(deepJSDM)
library(Hmsc)
load("data_sets.RData")

result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 10L)
auc = vector("list", nrow(setup))

.C("omp_set_num_threads_ptr", as.integer(6L))

set.seed(42,)


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
    
    
    # HMSC:
    hmsc = list()
    studyDesign = data.frame(sample = as.factor(1:nrow(train_Y)))
    rL = HmscRandomLevel(units = studyDesign$sample)
    model = Hmsc(Y = train_Y, XData = data.frame(train_X), XFormula = ~0 + .,
             studyDesign = studyDesign, ranLevels = list(sample = rL), distr = "probit")
    time =
      system.time({
        model = sampleMcmc(model, thin = 1, samples = 10000, transient = 1000,verbose = 5000,
                       nChains = 1L)
      })
    correlation = computeAssociations(model)[[1]]$mean
    species_weights = Hmsc::getPostEstimate(model,parName = "Beta")$mean
    
    
    
    
    
    result_corr_acc[i,j] =  sim$corr_acc(correlation)
    result_env[i,j] = mean(as.vector(species_weights > 0) == as.vector(sim$species_weights > 0))
    result_rmse_env[i,j] =  sqrt(mean((as.vector(species_weights) - as.vector(sim$species_weights))^2))
    result_time[i,j] = time[3]
    
    pred = Hmsc:::predict.Hmsc(model, XData = data.frame(test_X), type = "response")
    pred = apply(abind::abind(pred, along = -1L), 2:3, mean)
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
  saveRDS(hmsc, "results/hmsc.RDS")
}
