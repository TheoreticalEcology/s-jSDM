if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(deepJSDM)
library(gllvm)
library(BayesComm)
n = 3L
OpenMPController::omp_set_num_threads(n)
RhpcBLASctl::omp_set_num_threads(n)
RhpcBLASctl::blas_set_num_threads(n)
TMB::openmp(n = n)
seed = 42L

set.seed(seed)
.torch$manual_seed(seed)

sites = seq(20,by = 10, length.out = 25)
species = 20L
env = 3L


data_set = vector("list", 10L)
for(i in 1:length(sites)) {
  tmp = vector("list", 10L)
  for(j in 1:10){
    tmp[[j]] = simulate_SDM(env, sites = sites[i], species = species)
  }
  data_set[[i]] = tmp
}


useGPU(1L)
result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, length(sites),ncol = 10L)
for(i in 1:length(sites)) {
  for(j in 1:10){
    .torch$cuda$empty_cache()
    X = data_set[[i]][[j]]$env_weights
    Y = data_set[[i]][[j]]$response
    sim = data_set[[i]][[j]]
    
    model = createModel(X, Y)
    model = layer_dense(model, ncol(Y),FALSE, FALSE)
    model = compileModel(model, nLatent = 10L,lr = 1.0,optimizer = "LBFGS",reset = TRUE)
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
    #saveRDS(setup, file = "benchmark.RDS")
  }

}

gpu_behvaiour = list(
  result_corr_acc = result_corr_acc,
  result_env = result_env,
  result_rmse_env = result_rmse_env,
  result_time= result_time
)
saveRDS(gpu_behvaiour, "results/gpu_behvaiour_sites.RDS")

result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, length(sites),ncol = 10L)
for(i in 1:length(sites)) {
  for(j in 1:10){
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
  
gllvm_behvaiour = list(
    result_corr_acc = result_corr_acc,
    result_env = result_env,
    result_rmse_env = result_rmse_env,
    result_time= result_time
  )
saveRDS(gllvm_behvaiour, "results/gllvm_behvaiour_sites.RDS")


result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, length(sites),ncol = 10L)
for(i in 1:length(sites)) {
  for(j in 1:10){
    X = data_set[[i]][[j]]$env_weights
    Y = data_set[[i]][[j]]$response
    sim = data_set[[i]][[j]]
  
    time =
      system.time({
        model = BayesComm::BC(Y, X,model = "full", its = 10000)
      })
    
    cov = summary(model, "R")$statistics[,1]
    covFill = matrix(0,ncol(Y), ncol(Y))
    covFill[upper.tri(covFill)] = cov
    correlation = t(covFill)
    
    species_weights = matrix(NA, ncol(X), ncol(Y))
    n = paste0("B$sp",1:ncol(Y) )
    for(v in 1:ncol(Y)){
      smm = BayesComm:::summary.bayescomm(model, n[v])
      species_weights[,v]= smm$statistics[-1,1]
    }
    
    try({
      result_corr_acc[i,j] =  sim$corr_acc(correlation)
      result_env[i,j] = mean(as.vector(species_weights > 0) == as.vector(sim$species_weights > 0))
      result_rmse_env[i,j] =  sqrt(mean((as.vector(species_weights) - as.vector(sim$species_weights))^2))
      result_time[i,j] = time[3]
      rm(model)
      gc()
    })
  }
}

bc_behvaiour = list(
  result_corr_acc = result_corr_acc,
  result_env = result_env,
  result_rmse_env = result_rmse_env,
  result_time= result_time
)

saveRDS(bc_behvaiour, "results/bc_behvaiour_sites.RDS")
