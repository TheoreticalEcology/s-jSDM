if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(deepJSDM)
library(Hmsc)
load("data_sets.RData")

result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 10L)
auc = vector("list", nrow(setup))

OpenMPController::omp_set_num_threads(6L)
RhpcBLASctl::omp_set_num_threads(6L)
RhpcBLASctl::blas_set_num_threads(6L)

set.seed(42)


counter = 1
for(i in which(setup$env == 5L)) {
  sub_auc = vector("list", 10L)
  post = vector("list", 10)
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
        model = sampleMcmc(model, thin = 50, samples = 1000, transient = 50,verbose = 5000,
                       nChains = 2L)
      })
    
    post = convertToCodaObject(model)
    ess.beta = effectiveSize(post$Beta)
    psrf.beta = gelman.diag(post$Beta, multivariate=FALSE)$psrf
    
    ess.gamma = effectiveSize(post$Gamma)
    psrf.gamma = gelman.diag(post$Gamma, multivariate=FALSE)$psrf
    
    diag = list(post = post, ess.beta = ess.beta, psrf.beta = psrf.beta, ess.gamma = ess.gamma, psrf.gamma = psrf.gamma)
    
    
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
  }
  auc[[i]] = sub_auc
  post[[i]] = diag
  
  hmsc = list(
    setup = setup[i,],
    result_corr_acc = result_corr_acc,
    result_env = result_env,
    result_rmse_env = result_rmse_env,
    result_time= result_time,
    auc = auc,
    post = post
  )
  saveRDS(hmsc, "results/hmscDiag.RDS")
}
