library(deepJSDM)
library(gllvm)
n = 6L
OpenMPController::omp_set_num_threads(n)
RhpcBLASctl::omp_set_num_threads(n)
RhpcBLASctl::blas_set_num_threads(n)
TMB::openmp(n = n)
.torch$set_num_threads(n)

new_model = function(env, pa){
  model = createModel(as.matrix(env), as.matrix(pa))
  model = layer_dense(model, hidden = ncol(pa), FALSE, FALSE)
  return(model)
}

runtime_case = function(env, pa, batch_size = 200L, optimizer = "adamax"){
  
  if(optimizer == "adamax"){
    lr = 0.01
    epochs = 50L
  } else {
    lr = 1.0
    epochs = 8L
  }
  
  ## CPU
  useCPU()
  env_scaled = mlr::normalizeFeatures(env)
  model_cpu = new_model(env_scaled, pa)
  model_cpu = compileModel(model_cpu,nLatent = as.integer(ncol(pa)/2), lr = lr, optimizer = optimizer,reset = TRUE)
  time_cpu = system.time({model_cpu = deepJ(model_cpu, epochs = epochs, batch_size = batch_size)})
  cpu = 
    list(
      time = time_cpu[3],
      weights = model_cpu$raw_weights[[1]][[1]][[1]],
      cov = model_cpu$sigma()
    )
  rm(model_cpu)
  
  ## GPU
  useGPU(2L)
  model_gpu = new_model(ncol(env), ncol(pa))
  model_gpu = compileModel(model_gpu,nLatent = as.integer(ncol(pa)/2), lr = lr, optimizer = optimizer,reset = TRUE)
  time_gpu= system.time({model_gpu = deepJ(model_gpu, epochs = epochs, batch_size = batch_size)})
  gpu = 
    list(
      time = time_gpu[3],
      weights = model_gpu$raw_weights[[1]][[1]][[1]],
      cov = model_gpu$sigma()
    )
  rm(model_gpu)
  .torch$cuda$empty_cache()
  
  ## GLLVM
  time_gl = system.time({model_gl = gllvm::gllvm(pa, data.frame(env_scaled), family = binomial("probit"))})
  gl = 
    list(
      time = time_gl[3],
      weights = t(coef(model_gl)$Xcoef),
      cov = gllvm::getResidualCov(model_gl)$cov
    )
  rm(model_gl)
  return(results = list(
    cpu = cpu,
    gpu = gpu,
    gl = gl
  ))
}


# Birds
pa = read.csv("data/Birds/Birds_PA.csv")
coords = read.csv("data/Birds/Birds_LatLon.csv")
env = read.csv("data/Birds/Birds_Cov.csv")
bird_result = runtime_case(env, pa, 200L)


# Butterflies
pa = read.csv("data/Butterflies/Butterfly_PA.csv")
env = read.csv("data/Butterflies/Butterfly_Cov.csv")
butterflies_result = runtime_case(env, pa, 200L)


# Eucalypts
pa = read.csv("data/Eucalypts/Eucalypts_PA.csv")
env = read.csv("data/Eucalypts/Eucalypts_Covar.csv")
eucalypts_result = runtime_case(env, pa, nrow(env), optimizer = "LBFGS")


# Frogs
data = read.csv("data/Frogs/Anonymised_dataset.csv")
pa = data[,4:12]
env = data[,1:3]
frogs_result = runtime_case(env, pa, nrow(env), optimizer = "LBFGS")


# Fungi
data = read.csv("data/Fungi/Fungi_Compiled.csv")
pa = data[,1:11]
env = data[,12:ncol(data)]
fungi_result = runtime_case(env, pa, nrow(env), optimizer = "LBFGS")



# Mosquitoes
pa = read.csv("data/Mosquitos/Mosquito_PA.csv")
env = read.csv("data/Mosquitos/Mosquito_Covar.csv")
mosquitos_result = runtime_case(env, pa, nrow(env), optimizer = "LBFGS")


result = list(bird_result = bird_result, 
              butterflies_result = butterflies_result, 
              eucalypts_result = eucalypts_result, 
              frogs_result = frogs_result, 
              fungi_result = fungi_result,
              mosquitos_result = mosquitos_result)

saveRDS(file = "results/case_study_runtime.RDS")


