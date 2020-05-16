########## Default parameter #########
# epochs = 50L
# lr = 0.01
# batch_size = 10% of data
# nlatent = 50% of n species

if(version$minor > 5) RNGkind(sampleee.kind="Rounding")
library(sjSDM)
load("data_sets_sparse.RData")


result_corr_acc = result_corr_acc_min =result_corr_auc= result_corr_tss = result_time =  matrix(NA, nrow(setup),ncol = 10L)
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


torch$manual_seed(42L)
torch$cuda$manual_seed(42L)
set.seed(42)

lrs = seq(-12, -0.1, length.out = 18)
f = function(x) 2^x
lrs = f(lrs)

counter = 1
for(i in 1:nrow(setup)) {
  sub_auc = vector("list", 10L)
  for(j in 1:10){
    #.torch$cuda$empty_cache()
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

    model = sjSDM(train_Y, env=linear(train_X, ~0+.), iter = 100L, device = 2L, learning_rate = 0.003, 
                   #biotic = bioticStruct(lambda = best[["lambda_cov"]], alpha = best[["alpha_cov"]]))
                  biotic = bioticStruct(lambda = 0.1, alpha = 0.5))
    res = list(sigma = getCov(model), raw_weights = t(coef(model)[[1]]), pred = predict(model, test_X), 
               confusion = cf_function(round(getCov(model), 4), sim$correlation))
    
    result_corr_acc[i,j] =   sim$corr_acc(round(res$sigma, 4))
    result_corr_acc_min[i,j] =  NA
    
    Sens = res$confusion$cm$byClass[,1]
    Spec = res$confusion$cm$byClass[,2]
    TSS = Sens+ Spec - 1
    result_corr_tss[i,j] = sum(table(res$confusion$true)/sum(table(res$confusion$true))*TSS)
    
    result_corr_auc[i,j] = macro_auc(sim$correlation, round(res$sigma, 4))
    result_time[i,j] = model$time
    sub_auc[[j]] = list(pred = res, true = test_Y)
    gc()
    torch$cuda$empty_cache()
    #saveRDS(setup, file = "benchmark.RDS")
  }
  auc[[i]] = sub_auc

  gpu_dmvp = list(
    setup = setup[i,],
    result_corr_acc = result_corr_acc,
    result_corr_acc_min = result_corr_acc_min,
    result_time= result_time,
    result_corr_tss = result_corr_tss,
    result_corr_auc = result_corr_auc,
    auc = auc
  )
  saveRDS(gpu_dmvp, "results/sparse_gpu_sjSDM.RDS")
}


