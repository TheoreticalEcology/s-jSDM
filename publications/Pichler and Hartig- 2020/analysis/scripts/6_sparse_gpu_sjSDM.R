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

########## Parallel setup: ###########
library(snow)
cl = makeCluster(6L)
snow::clusterExport(cl, list("cf_function", "macro_auc"))
snow::clusterEvalQ(cl,library(deepJSDM) )

nodes = unlist(snow::clusterEvalQ(cl, paste(Sys.info()[['nodename']], Sys.getpid(), sep='-')))
snow::clusterExport(cl, list("nodes"))


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
    
    snow::clusterExport(cl, list("train_X", "train_Y", "test_X", "test_Y", "sim", "tmp"))
    time = system.time({
      res_tmp = parLapply(cl,lrs, function(lambda) {
        if(paste(Sys.info()[['nodename']], Sys.getpid(), sep='-') %in% nodes[1:3]) dev = 1L
        else dev = 2L
        torch$cuda$manual_seed(42L)
         
        model = sjSDM(train_X, train_Y, formula = ~0+X1+X2+X3+X4+X5, df = as.integer(tmp$species*tmp$sites), learning_rate = 0.01,
                      l1_cov = 0.5*lambda, l2_cov= 0.5*lambda, iter = 50L, batch_size = as.integer(nrow(train_X)*0.1), device = dev)
        res = list(sigma = getCov(model), raw_weights = coef(model), pred = predict(model, test_X), confusion = cf_function(round(getCov(model), 4), sim$correlation))
        rm(model)
        .torch$cuda$empty_cache()
        return(res)
      })
    })

    result_corr_acc[i,j] =  max(sapply(res_tmp, function(rr) sim$corr_acc(round(rr$sigma, 4))))
    result_corr_acc_min[i,j] =  min(sapply(res_tmp, function(rr) sim$corr_acc(round(rr$sigma,4))))
    result_corr_tss[i,j] = max(sapply(res_tmp, function(rr) {
      Sens = rr$confusion$cm$byClass[,1]
      Spec = rr$confusion$cm$byClass[,2]
      TSS = Sens+ Spec - 1
      return(sum(table(rr$confusion$true)/sum(table(rr$confusion$true))*TSS))
    }))
    
    result_corr_auc[i,j] = max(sapply(res_tmp, function(rr) {
      return(macro_auc(sim$correlation, round(rr$sigma, 4)))
    }))
    
    result_time[i,j] = time[3]
    sub_auc[[j]] = list(pred = res_tmp, true = test_Y)
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
  saveRDS(gpu_dmvp, "results/sparse_gpu_sjSDM.RDS")
}


