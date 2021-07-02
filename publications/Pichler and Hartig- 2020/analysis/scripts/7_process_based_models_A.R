library(Hmsc)
library(sjSDM)
library(BayesComm)
library(gllvm)
library(snow)
set.seed(42)
torch$manual_seed(42)
OpenMPController::omp_set_num_threads(4L)
RhpcBLASctl::omp_set_num_threads(4L)
RhpcBLASctl::blas_set_num_threads(4L)

sims = readRDS("data_process_based.rds")

E = readRDS("testingHMSC/manuscript_functions/fixedLandscapes/orig-no-seed-E.RDS")
MEMsel = readRDS("testingHMSC/manuscript_functions/fixedLandscapes/orig-no-seed-MEMsel.RDS")
X = cbind(scale(E),scale(E)^2, MEMsel)

cl = snow::makeCluster(5L)
snow::clusterEvalQ(cl, { library(Hmsc);library(gllvm);library(BayesComm);library(sjSDM);OpenMPController::omp_set_num_threads(4L);RhpcBLASctl::omp_set_num_threads(4L);RhpcBLASctl::blas_set_num_threads(4L) })
snow::clusterExport(cl, list("X", "sims"), envir = environment())

results = list("vector", 7)
for(i in 1:7) {
  snow::clusterExport(cl, list("i"), envir = environment())
  
  cat("Round: ", i, "\n")
  
  results[[i]] = snow::parLapply(cl, 1:5, function(j) {
    results = list()
    Y = sims$simulation[[i]][[j]]
    
    # # sjSDM
    # model = sjSDM(Y = Y, data.frame(X), iter = 100L, family = binomial("probit"), device = sample.int(3, 1)-1L)
    # results$sjSDM = list(beta = coef(model)[[1]], sjSDM::getCov(model))
    
    # Hmsc
    hmsc = list()
    studyDesign = data.frame(sample = as.factor(1:nrow(Y)))
    rL = HmscRandomLevel(units = studyDesign$sample)
    model = Hmsc(Y = Y, XData = data.frame(X), XFormula = ~1+.,
                 studyDesign = studyDesign, ranLevels = list(sample = rL), distr = "probit")
    model = sampleMcmc(model, thin = 50, samples = 1000, transient = 5000, verbose = 5000,
                       nChains = 2L)
    results$Hmsc = list(beta =  Hmsc::getPostEstimate(model,parName = "Beta")$mean, correlation = computeAssociations(model)[[1]]$mean)
    
    # gllvm
    error = tryCatch({
      time = system.time({
        model = gllvm::gllvm(y = Y, X = data.frame(X),formula = ~1+., family = binomial("probit"), starting.val = "zero", seed = 42)
      })},error = function(e) e)
    if("error"  %in% class(error)) {
      rm(error)
      error = tryCatch({
        time = system.time({
          model = gllvm::gllvm(y = Y, X = data.frame(X),formula = ~1+., family = binomial("probit"), starting.val = "random", seed = 42)
        })},error = function(e) e)
    }
    if("error"  %in% class(error)) {
      rm(error)
      error = tryCatch({
        time = system.time({
          model = gllvm::gllvm(y = Y, X = data.frame(X),formula = ~., family = binomial("probit"), seed = 42, num.lv = 1L)
        })},error = function(e) e)
    }
    try({results$gllvm = list(beta = coef(model)$Xcoef, correlation = gllvm::getResidualCov(model)$cov)}, silent = TRUE)
    
    # 
    # # BayesComm
    # model = BayesComm::BC(Y, as.matrix(X),model = "full", its = 50000, thin = 50, burn = 5000)
    # cov = summary(model, "R")$statistics[,1]
    # covFill = matrix(0,ncol(Y), ncol(Y))
    # covFill[upper.tri(covFill)] = cov
    # species_weights = matrix(NA, ncol(X)+1, ncol(Y))
    # n = paste0("B$sp",1:ncol(Y) )
    # for(v in 1:ncol(Y)){
    #   smm = BayesComm:::summary.bayescomm(model, n[v])
    #   species_weights[,v]= smm$statistics[,1]
    # }
    # results$BayesComm = list(beta = species_weights, correlation = t(covFill))
    return(results)
  })
  saveRDS(results, "results/7_process_based_gllvm_Hmsc.RDS")
}