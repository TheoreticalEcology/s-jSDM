########## Default parameter #########
# binomial with probit link
# with increasing number of species, nlatent -> 2 - 6

if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(deepJSDM)
library(gllvm)
load("data_sets_sparse.RData")
TMB::openmp(n = 6L)


result_corr_acc = result_corr_acc_min = result_corr_tss = result_time =  matrix(NA, nrow(setup),ncol = 10L)
auc = vector("list", nrow(setup))

cf_function = function(pred, true, threshold = 0.0){
  pred = pred[lower.tri(pred)]
  true = true[lower.tri(true)]
  pred = cut(pred, breaks = c(-1.0, -threshold- .Machine$double.eps, threshold+.Machine$double.eps, 1),labels = c("neg", "zero", "pos"))
  true = cut(true, breaks = c(-1.0, -threshold- .Machine$double.eps, threshold+.Machine$double.eps, 1),labels = c("neg", "zero", "pos"))
  return(list(cm = caret::confusionMatrix(pred, true), true = true, pred = pred))
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
    model = gllvm::gllvm(y = train_Y, X = data.frame(train_X), family = binomial("probit"), num.lv = dict[[as.character(tmp$species)]])
    })},error = function(e) e)
    if("error"  %in% class(error)) {
      rm(error)
      error = tryCatch({
        time = system.time({
          model = gllvm::gllvm(y = train_Y, X = data.frame(train_X), family = binomial("probit"), num.lv = dict[[as.character(tmp$species)]], starting.val = "zero")
        })},error = function(e) e)
    }
    if("error"  %in% class(error)) {
      rm(error)
      error = tryCatch({
        time = system.time({
          model = gllvm::gllvm(y = train_Y, X = data.frame(train_X), family = binomial("probit"), num.lv = dict[[as.character(tmp$species)]], starting.val = "random")
        })},error = function(e) e)
    }
    try({
    res = list(sigma = gllvm::getResidualCov(model)$cov, raw_weights = coef(model)$Xcoef, 
               pred = predict.gllvm(model, newX = data.frame(test_X), type = "response"), 
               confusion = cf_function(round(gllvm::getResidualCov(model)$cov, 4), sim$correlation))
      
    result_corr_acc[i,j] =  sim$corr_acc(gllvm::getResidualCov(model)$cov)
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
  
  gllvm = list(
    setup = setup[i,],
    result_corr_acc = result_corr_acc,
    result_time= result_time,
    result_corr_tss = result_corr_tss,
    auc = auc
  )
  saveRDS(gllvm, "results/sparse_gllvm.RDS")
}

