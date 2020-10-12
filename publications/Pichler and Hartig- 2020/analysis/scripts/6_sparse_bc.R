########## Default parameter #########
# binomial with probit link
# with increasing number of species, nlatent -> 2 - 6


if(version$minor > 5) RNGkind(sample.kind="Rounding")
#library(deepJSDM)
library(BayesComm)
load("data_sets_sparse_95.RData")


result_corr_acc = result_corr_acc2 = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 5L)
auc = diagnosis =vector("list", nrow(setup))


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
  
  for(j in 1:5L){
    
    tmp = data_sets[[counter]]$setup
    
    ### split into train and test ###
    train_X = data_sets[[counter]]$train_X
    train_Y = data_sets[[counter]]$train_Y
    test_X = data_sets[[counter]]$test_X
    test_Y = data_sets[[counter]]$test_Y
    sim = data_sets[[counter]]$sim
    
    
    # BayesComm:
    time =
      system.time({
        model1 = BayesComm::BC(train_Y, train_X,model = "full", its = 50000, thin = 50, burn = 5000)
        model2 = BayesComm::BC(train_Y, train_X,model = "full", its = 50000, thin = 50, burn = 5000)
      })
    
    cov = summary(model1, "R")$statistics[,1]
    covFill = matrix(0,ncol(train_Y), ncol(train_Y))
    covFill[upper.tri(covFill)] = cov
    correlation = t(covFill)
    
    species_weights = matrix(NA, ncol(train_X)+1, ncol(train_Y))
    n = paste0("B$sp",1:ncol(train_Y) )
    for(v in 1:ncol(train_Y)){
      smm = BayesComm:::summary.bayescomm(model1, n[v])
      species_weights[,v]= smm$statistics[,1]
    }
    
    m1 = lapply(model1$trace$B, function(mc) coda::as.mcmc(mc))
    m2 = lapply(model2$trace$B, function(mc) coda::as.mcmc(mc))
    beta.psrfs = lapply(1:length(model1$trace$B), function(i) coda::gelman.diag(coda::as.mcmc.list(list(m1[[i]], m2[[i]])),multivariate = FALSE)$psrf)
    beta.conv = abind::abind(beta.psrfs, along = 1L)[,1] > 1.2
    
    m1 = coda::as.mcmc(model1$trace$R)
    m2 = coda::as.mcmc(model2$trace$R)
    cov.psrf = coda::gelman.diag(coda::as.mcmc.list(list(m1, m2)),multivariate = FALSE)$psrf
    cov.conv = cov.psrf[,1] > 1.2
    
    diag = list(beta.conv = beta.conv , psrf.gamma = cov.conv, correlation = correlation)
    
    true_species_weights = rbind(rep(0.0, ncol(train_Y)), sim$species_weights)
    
    result_corr_acc[i,j] =  accuracy(sim$correlation, correlation)
    result_corr_acc2[i,j] =  accuracy2(sim$correlation, correlation)
    result_env[i,j] = mean(as.vector(species_weights[-1,] > 0) == as.vector(sim$species_weights > 0))
    result_rmse_env[i,j] =  sqrt(mean((as.vector(species_weights) - as.vector(true_species_weights))^2))
    result_time[i,j] = time[3]
    
    pred = BayesComm:::predict.bayescomm(model1, test_X)
    pred = apply(pred, 1:2, mean)
    sub_auc[[j]] = list(pred = pred, true = test_Y)
    post[[j]] = diag
    rm(model1)
    rm(model2)
    gc()
    counter = counter + 1L
  }
  auc[[i]] = sub_auc
  diagnosis[[i]] = post
  
  bc = list(
    setup = setup[i,],
    result_corr_acc = result_corr_acc,
    result_corr_acc2 = result_corr_acc2,
    result_env = result_env,
    result_rmse_env = result_rmse_env,
    result_time= result_time,
    auc = auc,
    post = diagnosis
  )
  saveRDS(bc, "results/6_BayesComm_sparse.RDS")
}
