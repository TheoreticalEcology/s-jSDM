if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(deepJSDM)
library(gllvm)
library(BayesComm)
library(Hmsc)
useGPU(2L)

n = 3L
OpenMPController::omp_set_num_threads(n)
RhpcBLASctl::omp_set_num_threads(n)
RhpcBLASctl::blas_set_num_threads(n)
TMB::openmp(n = n)
seed = 42L

set.seed(seed)
.torch$manual_seed(seed)

sites = seq(50,by = 20, length.out = 15)
species = 50L
env = 3L


data_set = vector("list", 15L)
for(i in 1:length(sites)) {
  tmp = vector("list", 5L)
  for(j in 1:5){
    tmp[[j]] = simulate_SDM(env, sites = sites[i], species = species)
  }
  data_set[[i]] = tmp
}


#### gpu dmvp ####
result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, length(sites),ncol = 5L)
for(i in 1:length(sites)) {
  for(j in 1:5){
    .torch$cuda$empty_cache()
    X = data_set[[i]][[j]]$env_weights
    Y = data_set[[i]][[j]]$response
    sim = data_set[[i]][[j]]
    
    model = createModel(X, Y)
    model = layer_dense(model, ncol(Y),FALSE, FALSE)
    model = compileModel(model, nLatent = 25L,lr = 0.02,optimizer = "adamax",reset = TRUE)
    time = system.time({
      model = deepJ(model, epochs = 50L,batch_size = as.integer(nrow(X)*0.1),corr = FALSE)
    })
    
    result_corr_acc[i,j] =  sim$corr_acc(model$sigma())
    result_env[i,j] = mean(as.vector(model$raw_weights[[1]][[1]][[1]] > 0) == as.vector(sim$species_weights > 0))
    result_rmse_env[i,j] =  sqrt(mean((as.vector(model$raw_weights[[1]][[1]][[1]]) - as.vector(sim$species_weights))^2))
    result_time[i,j] = time[3]
    rm(model)
    gc()
    .torch$cuda$empty_cache()
  }

}

gpu_behaviour = list(
  result_corr_acc = result_corr_acc,
  result_env = result_env,
  result_rmse_env = result_rmse_env,
  result_time= result_time
)
saveRDS(gpu_behaviour, "results/gpu_behaviour_sites_adamax.RDS")


result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, length(sites),ncol = 5L)
for(i in 1:length(sites)) {
  for(j in 1:5){
    .torch$cuda$empty_cache()
    X = data_set[[i]][[j]]$env_weights
    Y = data_set[[i]][[j]]$response
    sim = data_set[[i]][[j]]
    
    model = createModel(X, Y)
    model = layer_dense(model, ncol(Y),FALSE, FALSE)
    model = compileModel(model, nLatent = 25L,lr = 1.0,optimizer = "LBFGS",reset = TRUE)
    time = system.time({
      model = deepJ(model, epochs = 8L,batch_size = nrow(X),corr = FALSE)
    })
    
    result_corr_acc[i,j] =  sim$corr_acc(model$sigma())
    result_env[i,j] = mean(as.vector(model$raw_weights[[1]][[1]][[1]] > 0) == as.vector(sim$species_weights > 0))
    result_rmse_env[i,j] =  sqrt(mean((as.vector(model$raw_weights[[1]][[1]][[1]]) - as.vector(sim$species_weights))^2))
    result_time[i,j] = time[3]
    rm(model)
    gc()
    .torch$cuda$empty_cache()
  }
  
}

gpu_behaviour = list(
  result_corr_acc = result_corr_acc,
  result_env = result_env,
  result_rmse_env = result_rmse_env,
  result_time= result_time
)
saveRDS(gpu_behaviour, "results/gpu_behaviour_sites_lbfgs.RDS")



#### gllvm ####
result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, length(sites),ncol = 5L)
for(i in 1:length(sites)) {
  for(j in 1:5L){
    X = data_set[[i]][[j]]$env_weights
    Y = data_set[[i]][[j]]$response
    sim = data_set[[i]][[j]]
    
    error = tryCatch({
      time = system.time({
        model = gllvm::gllvm(y = Y, X = data.frame(X), family = binomial("probit"), num.lv = 2L, seed = seed)
      })},error = function(e) e)
    
    if("error"  %in% class(error)) {
      rm(error)
      error = tryCatch({
        time = system.time({
          model = gllvm::gllvm(y = Y, X = data.frame(X), family = binomial("probit"), num.lv = 2L, starting.val = "zero", seed = seed)
        })
      },error = function(e) e)
    }
    
    if("error"  %in% class(error)) {
      rm(error)
      error = tryCatch({
        time = system.time({
          model = gllvm::gllvm(y = Y, X = data.frame(X), family = binomial("probit"), num.lv = 2L, starting.val = "random", seed = seed)
        })},error = function(e) e)
    }
    try({
      result_corr_acc[i,j] =  sim$corr_acc(gllvm::getResidualCov(model)$cov)
      result_env[i,j] = mean(as.vector(t(coef(model)$Xcoef) > 0) == as.vector(sim$species_weights > 0))
      result_rmse_env[i,j] =  sqrt(mean((as.vector(t(coef(model)$Xcoef)) - as.vector(sim$species_weights))^2))
      result_time[i,j] = time[3]
    rm(model)
    gc()
    })
  }
}
  
gllvm_behaviour = list(
    result_corr_acc = result_corr_acc,
    result_env = result_env,
    result_rmse_env = result_rmse_env,
    result_time= result_time
  )
saveRDS(gllvm_behaviour, "results/gllvm_behaviour_sites.RDS")



#### bc ####
result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, length(sites),ncol = 5L)
posterior = vector("list", length(sites))
for(i in 1:length(sites)) {
  sub_posterior = vector("list", 5L)
  for(j in 1:5L){
    X = data_set[[i]][[j]]$env_weights
    Y = data_set[[i]][[j]]$response
    sim = data_set[[i]][[j]]
    try({
    time =
      system.time({
        model1 = BayesComm::BC(Y, X,model = "full", its = 50000, thin = 50, burn = 2500)
        model2 = BayesComm::BC(Y, X,model = "full", its = 50000, thin = 50, burn = 2500)
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
    
    try({
      result_corr_acc[i,j] =  sim$corr_acc(correlation)
      result_env[i,j] = mean(as.vector(species_weights > 0) == as.vector(sim$species_weights > 0))
      result_rmse_env[i,j] =  sqrt(mean((as.vector(species_weights) - as.vector(sim$species_weights))^2))
      result_time[i,j] = time[3]
      rm(model)
      gc()
      
      sub_posterior[[j]] = diag
      
    })
    }, silent = TRUE)
  }
  posterior[[i]] = sub_posterior
}


bc_behaviour = list(
  result_corr_acc = result_corr_acc,
  result_env = result_env,
  result_rmse_env = result_rmse_env,
  result_time= result_time,
  posterior = posterior
)

saveRDS(bc_behaviour, "results/bc_behaviour_sites.RDS")



#### hmsc ####
result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, length(sites),ncol = 5L)
posterior = vector("list", length(sites))
for(i in 1:length(sites)) {
  sub_posterior = vector("list", 5L)
  for(j in 1:5L){
    X = data_set[[i]][[j]]$env_weights
    Y = data_set[[i]][[j]]$response
    sim = data_set[[i]][[j]]
    
   
    # HMSC:
    studyDesign = data.frame(sample = as.factor(1:nrow(Y)))
    rL = HmscRandomLevel(units = studyDesign$sample)
    model = Hmsc(Y = Y, XData = data.frame(X), XFormula = ~0 + .,
                 studyDesign = studyDesign, ranLevels = list(sample = rL),distr = "probit")
    time =
      system.time({
        model = sampleMcmc(model, thin = 50, samples = 1000, transient = 50,verbose = 5000,
                           nChains = 2L,nParallel = 2L)
      })
    posterior = convertToCodaObject(model)
    ess.beta = effectiveSize(posterior$Beta)
    psrf.beta = gelman.diag(posterior$Beta, multivariate=FALSE)$psrf
    
    ess.gamma = effectiveSize(posterior$Gamma)
    psrf.gamma = gelman.diag(posterior$Gamma, multivariate=FALSE)$psrf
    
    diag = list(post = posterior, ess.beta = ess.beta, psrf.beta = psrf.beta, ess.gamma = ess.gamma, psrf.gamma = psrf.gamma)
    
    
    correlation = computeAssociations(model)[[1]]$mean
    species_weights = Hmsc::getPostEstimate(model,parName = "Beta")$mean
    
    
    try({
      result_corr_acc[i,j] =  sim$corr_acc(correlation)
      result_env[i,j] = mean(as.vector(species_weights > 0) == as.vector(sim$species_weights > 0))
      result_rmse_env[i,j] =  sqrt(mean((as.vector(species_weights) - as.vector(sim$species_weights))^2))
      result_time[i,j] = time[3]
      rm(model)
      gc()
      sub_posterior[[j]] = diag
    })
  }
  posterior[[i]] = sub_posterior
}

hmsc_behaviour = list(
  result_corr_acc = result_corr_acc,
  result_env = result_env,
  result_rmse_env = result_rmse_env,
  result_time= result_time,
  posterior = posterior
)

saveRDS(hmsc_behaviour, "results/hmsc_behaviour_sites.RDS")