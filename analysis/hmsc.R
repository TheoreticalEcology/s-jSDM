if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(deepJSDM)
library(Hmsc)
sites = c(50, 70, 100, 140, 180, 260, 320, 400, 500)[1:5]
species = c(0.1, 0.2, 0.3, 0.4,0.5)
env = c(3,5,7)


setup = expand.grid(sites, species, env)
colnames(setup) = c("sites", "species", "env")
setup = setup[order(setup$sites,decreasing = FALSE),]
result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 10L)
auc = vector("list", nrow(setup))



set.seed(42,)



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
    
    # HMSC:
    hmsc = list()
    studyDesign = data.frame(sample = as.factor(1:nrow(train_Y)))
    rL = HmscRandomLevel(units = studyDesign$sample)
    model = Hmsc(Y = train_Y, XData = data.frame(train_X), XFormula = ~0 + .,
             studyDesign = studyDesign, ranLevels = list(sample = rL), distr = "probit")
    time =
      system.time({
        model = sampleMcmc(model, thin = 1, samples = 10000, transient = 1000,verbose = 100,
                       nChains = 1L)
      })
    correlation = computeAssociations(model)[[1]]$mean
    species_weights = Hmsc::getPostEstimate(model,parName = "Beta")$mean
    
    
    
    
    
    result_corr_acc[i,j] =  sim$corr_acc(correlation)
    result_env[i,j] = mean(as.vector(species_weights > 0) == as.vector(sim$species_weights > 0))
    result_rmse_env[i,j] =  sqrt(mean((as.vector(species_weights) - as.vector(sim$species_weights))^2))
    result_time[i,j] = time[3]
    
    pred = Hmsc:::predict.Hmsc(model, XData = data.frame(test_X), type = "response")
    sub_auc[[j]] = list(pred = pred, true = test_Y)
    rm(model)
    gc()
    .torch$cuda$empty_cache()
    #saveRDS(setup, file = "benchmark.RDS")
  }
  auc[[i]] = sub_auc
  
  hmsc = list(
    setup = setup,
    result_corr_acc = result_corr_acc,
    result_env = result_env,
    result_rmse_env = result_rmse_env,
    result_time= result_time,
    auc = auc
  )
  saveRDS(hmsc, "results/hmsc.RDS")
}
