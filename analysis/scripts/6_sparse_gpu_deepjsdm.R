########## Default parameter #########
# epochs = 50L
# lr = 0.01
# batch_size = 10% of data
# nlatent = 50% of n species

if(version$minor > 5) RNGkind(sampleee.kind="Rounding")
library(deepJSDM)
load("data_sets_sparse.RData")


result_corr_acc = result_corr_acc_min =  result_time =  matrix(NA, nrow(setup),ncol = 10L)
auc = vector("list", nrow(setup))



useGPU(2L)
.torch$manual_seed(42L)
.torch$cuda$manual_seed(42L)
set.seed(42)

lrs = seq(-12, -1, length.out = 12)
f = function(x) 2^x
lrs = f(lrs)

library(snow)
cl = makeCluster(4L)
#snow::clusterExport(cl, list("data_sets"))
snow::clusterEvalQ(cl,library(deepJSDM) )
snow::clusterEvalQ(cl,useGPU(2L))
snow::clusterEvalQ(cl,.torch$cuda$manual_seed(42L))

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
  
        model = createModel(train_X, train_Y)
        model = layer_dense(model,ncol(train_Y),FALSE, FALSE, l1 = 0.5*lambda, l2 = 0.5*lambda)
        model = compileModel(model, nLatent = as.integer(tmp$species*tmp$sites*0.5),lr = 0.01,optimizer = "adamax",reset = TRUE, l1 = 0.5*lambda, l2 = 0.5*lambda, reg_on_Cov = FALSE)
        model = deepJ(model, epochs = 50L,batch_size = as.integer(nrow(train_X)*0.1),corr = FALSE)
        res = list(sigma = model$sigma(), raw_weights = model$raw_weights, pred = predict(model, test_X))
        rm(model)
        .torch$cuda$empty_cache()
        return(res)
      })
    })

    result_corr_acc[i,j] =  max(sapply(res_tmp, function(rr) sim$corr_acc(round(rr$sigma, 4))))
    result_corr_acc_min[i,j] =  min(sapply(res_tmp, function(rr) sim$corr_acc(round(rr$sigma,4))))
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
    auc = auc
  )
  saveRDS(gpu_dmvp, "results/sparse_gpu_dmvp3.RDS")
}


