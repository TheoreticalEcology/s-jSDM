library(BayesComm)
OpenMPController::omp_set_num_threads(6L)
RhpcBLASctl::omp_set_num_threads(6L)
RhpcBLASctl::blas_set_num_threads(6L)
set.seed(42)


runtime_case = function(env, pa){
  time = system.time({m = BayesComm::BC(as.matrix(pa), as.matrix(env),model = "full", its = 11000,  burn = 1000, thin = 10)})
  return(time[3])
}


# Birds
pa = read.csv("data/Birds/Birds_PA.csv")
coords = read.csv("data/Birds/Birds_LatLon.csv")
env = read.csv("data/Birds/Birds_Cov.csv")
bird_result = vector("list",3)
for(i in 1:3) bird_result[[i]] = runtime_case(env, pa)


# Butterflies
pa = read.csv("data/Butterflies/Butterfly_PA.csv")
env = read.csv("data/Butterflies/Butterfly_Cov.csv")
butterflies_result = vector("list", 3)
for(i in 1:3) butterflies_result[[i]] = runtime_case(env, pa)


# Eucalypts
pa = read.csv("data/Eucalypts/Eucalypts_PA.csv")
env = read.csv("data/Eucalypts/Eucalypts_Covar.csv")
eucalypts_result = vector("list", 3)
for(i in 1:3) eucalypts_result[[i]] = runtime_case(env, pa)


# Frogs
data = read.csv("data/Frogs/Anonymised_dataset.csv")
pa = data[,4:12]
env = data[,1:3]
frogs_result = vector("list", 3)
for(i in 1:3) frogs_result[[i]] = runtime_case(env, pa)


# Fungi
data = read.csv("data/Fungi/Fungi_Compiled.csv")
pa = data[,1:11]
env = data[,12:ncol(data)]
fungi_result = vector("list", 3)
for(i in 1:3) fungi_result[[i]] = runtime_case(env, pa)



# Mosquitoes
pa = read.csv("data/Mosquitos/Mosquito_PA.csv")
env = read.csv("data/Mosquitos/Mosquito_Covar.csv")
mosquitos_result = vector("list", 3)
for(i in 1:3) mosquitos_result[[i]] = runtime_case(env, pa)


result = list(bird_result = bird_result, 
              butterflies_result = butterflies_result, 
              eucalypts_result = eucalypts_result, 
              frogs_result = frogs_result, 
              fungi_result = fungi_result,
              mosquitos_result = mosquitos_result)

saveRDS(result, file = "results/case_study_runtime_BC.RDS")


