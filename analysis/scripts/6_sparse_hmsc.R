########## Default parameter #########
# binomial with probit link
# with increasing number of species, nlatent -> 2 - 6

if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(deepJSDM)
library(Hmsc)
library(coda)
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
    
    try({
    hmsc = list()
    studyDesign = data.frame(sample = as.factor(1:nrow(train_Y)))
    rL = HmscRandomLevel(units = studyDesign$sample)
    model = Hmsc(Y = train_Y, XData = data.frame(train_X), XFormula = ~0 + .,
                 studyDesign = studyDesign, ranLevels = list(sample = rL), distr = "probit")
    time =
      system.time({
        model = sampleMcmc(model, thin = 50, samples = 1000, transient = 50,verbose = 5000,
                           nChains = 2L)
      })
    
    posterior = convertToCodaObject(model)
    ess.beta = effectiveSize(posterior$Beta)
    psrf.beta = gelman.diag(posterior$Beta, multivariate=FALSE)$psrf
    
    ess.gamma = effectiveSize(posterior$Gamma)
    psrf.gamma = gelman.diag(posterior$Gamma, multivariate=FALSE)$psrf
    
    diag = list(post = posterior, ess.beta = ess.beta, psrf.beta = psrf.beta, ess.gamma = ess.gamma, psrf.gamma = psrf.gamma)
    
    
    correlation = computeAssociations(model)[[1]]$mean
    species_weights = Hmsc::getPostEstimate(model,parName = "Beta")$mean
    
    
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
    rm(model)
    gc()
    .torch$cuda$empty_cache()
    },silent = TRUE)
  }
  auc[[i]] = sub_auc
  
  hmsc = list(
    setup = setup[i,],
    result_corr_acc = result_corr_acc,
    result_time= result_time,
    result_corr_tss = result_corr_tss,
    result_corr_auc = result_corr_auc,
    auc = auc
  )
  saveRDS(hmsc, "results/sparse_hmsc.RDS")
}

