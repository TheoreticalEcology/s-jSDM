########## Default parameter #########
# epochs = 50L
# lr = 0.01
# batch_size = 10% of data
# nlatent = 50% of n species

if(version$minor > 5) RNGkind(sampleee.kind="Rounding")
library(deepJSDM)
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


useGPU(0L)
.torch$manual_seed(42L)
.torch$cuda$manual_seed(42L)
set.seed(42)





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
    
    time = system.time({
  
        model = createModel(train_X, train_Y)
        model = layer_dense(model,ncol(train_Y),FALSE, FALSE)
        model = compileModel(model, nLatent = as.integer(tmp$species*tmp$sites),lr = 0.01,optimizer = "adamax",reset = TRUE)
        model = deepJ(model, epochs = 50L,batch_size = as.integer(nrow(train_X)*0.1),corr = FALSE)
        res = list(sigma = model$sigma(), raw_weights = model$raw_weights, pred = predict(model, test_X), confusion = cf_function(round(model$sigma(), 4), sim$correlation))
        rm(model)
        .torch$cuda$empty_cache()
    })

    result_corr_acc[i,j] =  sim$corr_acc(round(res$sigma, 4))
   # result_corr_acc_min[i,j] =  min(sapply(res_tmp, function(rr) sim$corr_acc(round(rr$sigma,4))))
      Sens = res$confusion$cm$byClass[,1]
      Spec = res$confusion$cm$byClass[,2]
        
      TSS = Sens+ Spec - 1
      result_corr_tss[i,j] =sum(table(res$confusion$true)/sum(table(res$confusion$true))*TSS)
      
    
    result_corr_auc[i,j] = macro_auc(sim$correlation, round(res$sigma, 4))
    
    result_time[i,j] = time[3]
    sub_auc[[j]] = list(pred = res, true = test_Y)
    gc()
    .torch$cuda$empty_cache()
    
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
  saveRDS(gpu_dmvp, "results/sparse_gpu_dmvp_base.RDS")
}


