library(sjSDM)
torch$cuda$manual_seed(42L)
torch$manual_seed(42L)
n = 6L


runtime_case = function(env, pa, batch_size = 200L, optimizer = "adamax"){
  
  if(optimizer == "adamax"){
    learning_rate = 0.01
    epochs = 50L
  } else {
    learning_rate = 1.0
    epochs = 8L
  }
  pa = as.matrix(pa)
  ## CPU
  torch$set_num_threads(6L)
  fa$utils_fa$torch$set_num_threads(6L)
  
  env_scaled = mlr::normalizeFeatures(env)
  
  
  model = sjSDM(env_scaled, pa, formula = ~., learning_rate = learning_rate, 
                df = as.integer(ncol(pa)/2),iter = epochs, step_size = batch_size,
                device = "cpu")
  time = model$time
  cpu = 
    list(
      time = model$time,
      weights = coef(model)[[1]],
      cov = getCov(model)
    )
  rm(model)
  
  ## GPU
  torch$set_num_threads(3L)
  env_scaled = mlr::normalizeFeatures(env)
  
  
  model = sjSDM(env_scaled, pa, formula = ~., learning_rate = learning_rate, 
                df = as.integer(ncol(pa)/2),iter = epochs, step_size = batch_size,
                device = 0L)
  time = model$time
  gpu = 
    list(
      time = model$time,
      weights = coef(model)[[1]],
      cov = getCov(model)
    )
  rm(model)
  .torch$cuda$empty_cache()
  return(list(cpu = cpu, gpu = gpu))
}


# Birds
pa = read.csv("data/Birds/Birds_PA.csv")
coords = read.csv("data/Birds/Birds_LatLon.csv")
env = read.csv("data/Birds/Birds_Cov.csv")
bird_result = vector("list",10)
for(i in 1:10) bird_result[[i]] = runtime_case(env, pa, 200L)


# Butterflies
pa = read.csv("data/Butterflies/Butterfly_PA.csv")
env = read.csv("data/Butterflies/Butterfly_Cov.csv")
butterflies_result = vector("list", 10)
for(i in 1:10) butterflies_result[[i]] = runtime_case(env, pa, 200L)


# Eucalypts
pa = read.csv("data/Eucalypts/Eucalypts_PA.csv")
env = read.csv("data/Eucalypts/Eucalypts_Covar.csv")
eucalypts_result = vector("list", 10)
for(i in 1:10) eucalypts_result[[i]] = runtime_case(env, pa, nrow(env))


# Frogs
data = read.csv("data/Frogs/Anonymised_dataset.csv")
pa = data[,4:12]
env = data[,1:3]
frogs_result = vector("list", 10)
for(i in 1:10) frogs_result[[i]] = runtime_case(env, pa, nrow(env))


# Fungi
data = read.csv("data/Fungi/Fungi_Compiled.csv")
pa = data[,1:11]
env = data[,12:ncol(data)]
fungi_result = vector("list", 10)
for(i in 1:10) fungi_result[[i]] = runtime_case(env, pa, nrow(env))



# Mosquitoes
pa = read.csv("data/Mosquitos/Mosquito_PA.csv")
env = read.csv("data/Mosquitos/Mosquito_Covar.csv")
mosquitos_result = vector("list", 10)
for(i in 1:10) mosquitos_result[[i]] = runtime_case(env, pa, nrow(env))


result = list(bird_result = bird_result, 
              butterflies_result = butterflies_result, 
              eucalypts_result = eucalypts_result, 
              frogs_result = frogs_result, 
              fungi_result = fungi_result,
              mosquitos_result = mosquitos_result)

saveRDS(result, file = "results/case_study_runtime.RDS")


