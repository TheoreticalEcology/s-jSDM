########## Default parameter #########
# epochs = 50L
# lr = 0.01
# batch_size = 10% of data
# nlatent = 50% of n species

if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(sjSDM)
load("data_sets_sparse_95.RData")

result_corr_acc = result_corr_acc2 = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 5L)
auc = vector("list", nrow(setup))
diagnosis = vector("list", nrow(setup))

torch$set_num_threads(6L)
torch$manual_seed(42L)
torch$cuda$manual_seed(42L)
set.seed(42)
accuracy = function(true, pred, t = 0.01) {
  true = true[lower.tri(true)]
  pred = pred[lower.tri(pred)]
  zero_acc = mean((abs(true) < t) == (abs(pred) < t))
  
  if(any(true>t)) {
    pos_acc = mean((true>t)[(true>t)] == (pred > t)[(true>t)])
  } else {
    pos_acc = NULL
  }
  
  if(any(true< -t)){
    neg_acc =  mean((true< -t)[(true< -t)] == (pred < -t)[(true< -t)])
  } else {
    neg_acc = NULL
  }
  return(mean(c( zero_acc, pos_acc, neg_acc)))
}

parse_cov = function(cov, t){
  cov[abs(cov) < t] = 0
  cov[cov > t] = 1
  cov[cov < -t] = -1
  return(cov)
}

accuracy2 = function(true, pred, t = 0.01){
  true = parse_cov(true, t)
  pred = parse_cov(pred, t)
  return( mean(true[lower.tri(true)] == pred[lower.tri(pred)]) )
}


counter = 1
for(i in 1:nrow(setup)) {
  sub_auc = vector("list", 5L)
  post = vector("list", 5L)
  for(j in 1:5){
    torch$cuda$empty_cache()
    tmp = data_sets[[counter]]$setup
    ### split into train and test ###
    train_X = data_sets[[counter]]$train_X
    train_Y = data_sets[[counter]]$train_Y
    test_X = data_sets[[counter]]$test_X
    test_Y = data_sets[[counter]]$test_Y
    sim = data_sets[[counter]]$sim
    counter = counter + 1L
    
    tune = sjSDM_cv(train_Y, train_X, n_cores = 9L, n_gpu = 3, tune_steps = 40L, sampling = 2000L, 
                    iter = 50L, alpha_spatial=0, lambda_spatial=0, alpha_coef = 0.0, lambda_coef = 0.0, learning_rate=0.01, CV = 5L, 
                    lambda_cov = 2^seq(-10, -4, length.out = 20), alpha_cov = scales::rescale( 2^seq(-10, -4, length.out = 20) ), link="logit")
    best = head(tune$short_summary[order(tune$short_summary$logLik),])[1,]
    
    model = sjSDM(train_Y, env = linear(train_X, ~.), learning_rate = 0.01,
                  iter = 50L, link="logit",
                  device = 0L,
                  sampling = 2000L,
                  biotic = bioticStruct(lambda =best$lambda_cov, alpha = best$alpha_cov))
    
    time = model$time
    true_species_weights = rbind(rep(0.0, ncol(train_Y)), sim$species_weights)
    result_corr_acc[i,j] =  accuracy(sim$correlation, getCov(model))
    result_corr_acc2[i,j] =  accuracy2(sim$correlation, getCov(model))
    ce = t(coef(model)[[1]])
    result_env[i,j] = mean(as.vector(ce[-1,] > 0) == as.vector(sim$species_weights > 0))
    result_rmse_env[i,j] =  sqrt(mean((as.vector(ce) - as.vector(true_species_weights))^2))
    result_time[i,j] = time
    pred = apply(abind::abind(lapply(1:100, function(i) predict(model, newdata = test_X)), along = -1L), 2:3, mean)
    sub_auc[[j]] = list(pred = pred, true = test_Y)
    post[[j]] = list(correlation=getCov(model))
    rm(model)
    gc()
    torch$cuda$empty_cache()
    #saveRDS(setup, file = "benchmark.RDS")
  }
  auc[[i]] = sub_auc
  diagnosis[[i]] = post
  
  gpu_dmvp = list(
    setup = setup[i,],
    result_corr_acc = result_corr_acc,
    result_corr_acc2 = result_corr_acc2,
    result_env = result_env,
    result_rmse_env = result_rmse_env,
    result_time= result_time,
    auc = auc,
    post = diagnosis
  )
  saveRDS(gpu_dmvp, "results/6_gpu_sjSDM_sparse2.RDS")
}


