########## Default parameter #########
# binomial with probit link
# with increasing number of species, nlatent -> 2 - 6

if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(Hmsc)
load("data_sets_sparse_95.RData")

result_corr_acc = result_corr_acc2 = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 5L)
auc = vector("list", nrow(setup))
diagnosis = vector("list", nrow(setup))


OpenMPController::omp_set_num_threads(6L)
RhpcBLASctl::omp_set_num_threads(6L)
RhpcBLASctl::blas_set_num_threads(6L)
set.seed(42)

accuracy = function(true, pred, t = 0.01) {
  true = true[lower.tri(true)]
  pred = pred[lower.tri(pred)]
  zero_acc = mean((abs(true) < t) == (abs(pred) < t))
  
  if(any(true>t)) {
    pos_acc = mean((true>t)[(true>t)] == (pred > t)[(true>t)])
  } else {
    pos_acc = NULL
  }
  
  if(any(true< -t)){
    neg_acc =  mean((true< -t)[(true< -t)] == (pred < -t)[(true< -t)])
  } else {
    neg_acc = NULL
  }
  return(mean(c( zero_acc, pos_acc, neg_acc)))
}

parse_cov = function(cov, t){
  cov[abs(cov) < t] = 0
  cov[cov > t] = 1
  cov[cov < -t] = -1
  return(cov)
}

accuracy2 = function(true, pred, t = 0.01){
  true = parse_cov(true, t)
  pred = parse_cov(pred, t)
  return( mean(true[lower.tri(true)] == pred[lower.tri(pred)]) )
}


counter = 1
for(i in 1:nrow(setup)) {
  sub_auc = vector("list", 5L)
  post = vector("list", 5L)
  
  
  for(j in 1:5){
    
    tmp = data_sets[[counter]]$setup
    
    ### split into train and test ###
    train_X = data_sets[[counter]]$train_X
    train_Y = data_sets[[counter]]$train_Y
    test_X = data_sets[[counter]]$test_X
    test_Y = data_sets[[counter]]$test_Y
    sim = data_sets[[counter]]$sim
    
    # HMSC:
    hmsc = list()
    studyDesign = data.frame(sample = as.factor(1:nrow(train_Y)))
    rL = HmscRandomLevel(units = studyDesign$sample)
    model = Hmsc(Y = train_Y, XData = data.frame(train_X), XFormula = ~1+.,
                 studyDesign = studyDesign, ranLevels = list(sample = rL), distr = "probit")
    time =
      system.time({
        model = sampleMcmc(model, thin = 50, samples = 1000, transient = 5000, verbose = 5000,
                           nChains = 2L)
      })
    
    posterior = convertToCodaObject(model)
    ess.beta = effectiveSize(posterior$Beta)
    psrf.betas = gelman.diag(posterior$Beta, multivariate=FALSE)$psrf
    beta.conv = abind::abind(psrf.betas, along = 1L)[,1] > 1.2
    
    ess.gamma = effectiveSize(posterior$Gamma)
    psrf.gamma = gelman.diag(posterior$Gamma, multivariate=FALSE)$psrf
    cov.conv = psrf.gamma[,1] > 1.2
    
    ess.lambda = effectiveSize(posterior$Lambda[[1]])
    psrf.lambda = gelman.diag(posterior$Lambda[[1]], multivariate=FALSE)$psrf
    lambda.conv = psrf.lambda[,1] > 1.2
    
    OmegaCor = computeAssociations(model)
    supportLevel = 0.9
    correlation = ((OmegaCor[[1]]$support>supportLevel) + (OmegaCor[[1]]$support<(1-supportLevel))>0)*OmegaCor[[1]]$mean
    
    
    diag = list(beta.conv = beta.conv , psrf.gamma = cov.conv, correlation = OmegaCor, lambda.conv=lambda.conv)
    species_weights = Hmsc::getPostEstimate(model,parName = "Beta")$mean
    
    true_species_weights = rbind(rep(0.0, ncol(train_Y)), sim$species_weights)
    result_corr_acc[i,j] =  accuracy(sim$correlation, correlation)
    result_corr_acc2[i,j] =  accuracy2(sim$correlation, correlation)
    result_env[i,j] = mean(as.vector(species_weights[-1,] > 0) == as.vector(sim$species_weights > 0))
    result_rmse_env[i,j] =  sqrt(mean((as.vector(species_weights) - as.vector(true_species_weights))^2))
    result_time[i,j] = time[3]
    
    studyDesign = data.frame(sample = as.factor(1:nrow(test_X) + nrow(train_X) ))
    pred = Hmsc:::predict.Hmsc(model, XData = data.frame(test_X), type = "response", studyDesign = studyDesign,  expected = TRUE)
    pred = apply(abind::abind(pred, along = -1L), 2:3, mean)
    sub_auc[[j]] = list(pred = pred, true = test_Y)
    post[[j]] = diag
    rm(model)
    gc()
    counter = counter + 1L
  }
  auc[[i]] = sub_auc
  diagnosis[[i]] = post
  
  hmsc = list(
    setup = setup[i,],
    result_corr_acc = result_corr_acc,
    result_corr_acc2 = result_corr_acc2,
    result_env = result_env,
    result_rmse_env = result_rmse_env,
    result_time= result_time,
    auc = auc,
    post = diagnosis
  )
  saveRDS(hmsc, "results/6_hmsc_sparse.RDS")
}