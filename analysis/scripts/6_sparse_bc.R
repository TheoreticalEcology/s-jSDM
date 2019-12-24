########## Default parameter #########
# binomial with probit link
# with increasing number of species, nlatent -> 2 - 6

if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(deepJSDM)
library(BayesComm)
load("data_sets_sparse.RData")
TMB::openmp(n = 3L)

OpenMPController::omp_set_num_threads(3L)
RhpcBLASctl::omp_set_num_threads(3L)
RhpcBLASctl::blas_set_num_threads(3L)



result_corr_acc =result_corr_auc = result_corr_acc_min = result_corr_tss = result_time =  matrix(NA, nrow(setup),ncol = 10L)
auc = vector("list", nrow(setup))

cf_function = function(pred, true, threshold = 0.0){
  pred = pred[lower.tri(pred)]
  true = true[lower.tri(true)]
  pred = cut(pred, breaks = c(-1.0, -threshold- .Machine$double.eps, threshold+.Machine$double.eps, 1),labels = c("neg", "zero", "pos"))
  true = cut(true, breaks = c(-1.0, -threshold- .Machine$double.eps, threshold+.Machine$double.eps, 1),labels = c("neg", "zero", "pos"))
  return(list(cm = caret::confusionMatrix(pred, true), true = true, pred = pred))
}

macro_auc = function(true, pred) {
  cf =  cf_function(pred, true)
  zero = pos = neg =cf$true
  
  levels(zero) = c("1", "0", "1")
  levels(pos) = c("0", "0", "1")
  levels(neg) = c("1", "0", "0")
  
  pZ = abs(cov2cor(pred))[lower.tri(pred)]
  pP = scales::rescale(cov2cor(pred),to = c(0,1))[lower.tri(pred)]
  
  zero = as.numeric(as.character(zero))
  pos = as.numeric(as.character(pos))
  neg = as.numeric(as.character(neg))
  
  Metrics::auc(zero, pZ)
  Metrics::auc(pos, pP)
  Metrics::auc(neg, 1-pP)
  return(
    sum(table(cf$true)/sum(table(cf$true))*c(Metrics::auc(zero, pZ), Metrics::auc(pos, pP)
                                             , Metrics::auc(neg, 1-pP)))
  )
}


set.seed(42)

dict = as.list(2:6)
names(dict) = as.character(unique(setup$species))

counter = 1
for(i in 1:nrow(setup)) {
  sub_auc = vector("list", 10L)
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
    
    error = tryCatch({
    time = system.time({
      model1 = BayesComm::BC(train_Y, train_X,model = "full", its = 50000, thin = 50, burn = 5000)
      model2 = BayesComm::BC(train_Y, train_X,model = "full", its = 50000, thin = 50, burn = 5000)
    })},error = function(e) e)
    
    try({
      
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
    
    res = list(sigma = correlation, raw_weights = species_weights, 
               pred = BayesComm:::predict.bayescomm(model1, test_X), 
               confusion = cf_function(round(correlation, 4), sim$correlation),
               posterior = diag)
      
    result_corr_acc[i,j] =  sim$corr_acc(correlation)
    result_corr_auc[i,j] =  macro_auc(sim$correlation, round(correlation, 4))
    
    result_time[i,j] = time[3]
    
    Sens = res$confusion$cm$byClass[,1]
    Spec = res$confusion$cm$byClass[,2]
    TSS = Sens+ Spec - 1
    result_corr_tss[i,j] = sum(table(res$confusion$true)/sum(table(res$confusion$true))*TSS)
    
    sub_auc[[j]] = list(pred = res, true = test_Y)
    rm(model1, model2)
    gc()
    .torch$cuda$empty_cache()
    },silent = TRUE)
  }
  auc[[i]] = sub_auc
  
  bc = list(
    setup = setup[i,],
    result_corr_acc = result_corr_acc,
    result_time= result_time,
    result_corr_tss = result_corr_tss,
    result_corr_auc = result_corr_auc,
    auc = auc
  )
  saveRDS(bc, "results/sparse_bc.RDS")
}

