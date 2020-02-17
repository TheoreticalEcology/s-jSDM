########## Default parameter #########
# iterations = 50000
# burnin = 5000
# nchains = 2
# thin = 50

if(version$minor > 5) RNGkind(sample.kind="Rounding")
#library(deepJSDM)
library(BayesComm)
load("data_sets2.RData")


result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 10L)
auc = diagnosis =vector("list", nrow(setup))


OpenMPController::omp_set_num_threads(6L)
RhpcBLASctl::omp_set_num_threads(6L)
RhpcBLASctl::blas_set_num_threads(6L)
set.seed(42)


counter = 1
for(i in 1:nrow(setup)) {
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
      
      species_weights = matrix(NA, ncol(train_X), ncol(train_Y))
      n = paste0("B$sp",1:ncol(train_Y) )
      for(v in 1:ncol(train_Y)){
        smm = BayesComm:::summary.bayescomm(model1, n[v])
        species_weights[,v]= smm$statistics[-1,1]
      }
      
      m1 = lapply(model1$trace$B, function(mc) coda::as.mcmc(mc))
      m2 = lapply(model2$trace$B, function(mc) coda::as.mcmc(mc))
      beta.psrfs = lapply(1:length(model1$trace$B), function(i) coda::gelman.diag(coda::as.mcmc.list(list(m1[[i]], m2[[i]])),multivariate = FALSE)$psrf)
      
      
      m1 = coda::as.mcmc(model1$trace$R)
      m2 = coda::as.mcmc(model2$trace$R)
      cov.psrf = coda::gelman.diag(coda::as.mcmc.list(list(m1, m2)),multivariate = FALSE)$psrf
      
      diag = list(post = list(m1 = m1, m2 = m2), psrf.beta = beta.psrfs, psrf.gamma = cov.psrf)
      
      
      result_corr_acc[i,j] =  sim$corr_acc(correlation)
      result_env[i,j] = mean(as.vector(species_weights > 0) == as.vector(sim$species_weights > 0))
      result_rmse_env[i,j] =  sqrt(mean((as.vector(species_weights) - as.vector(sim$species_weights))^2))
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
    result_env = result_env,
    result_rmse_env = result_rmse_env,
    result_time= result_time,
    auc = auc,
    post = diagnosis
  )
  saveRDS(bc, "results/BayesCommDiag.RDS")
}
