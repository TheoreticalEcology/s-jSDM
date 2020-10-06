########## Default parameter #########
# binomial with probit link
# with increasing number of species, nlatent -> 2 - 6

if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(gllvm)
load("data_sets_full.RData")
TMB::openmp(n = 6L)

result_corr_acc = result_env = result_rmse_env =  result_time =  matrix(NA, nrow(setup),ncol = 5L)
auc = vector("list", nrow(setup))
diagnosis = vector("list", nrow(setup))

set.seed(42)

dict = as.list(2:6)
names(dict) = as.character(unique(setup$species))

counter = 1
for(i in 1:nrow(setup)) {
  sub_auc = vector("list", 5L)
  post = vector("list", 5L)
  for(j in 1:5L){
    
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
    model = gllvm::gllvm(y = train_Y, X = data.frame(train_X),formula = ~X1+X2+X3+X4+X5, family = binomial("probit"), num.lv = dict[[as.character(tmp$species)]], seed = 42)
    })},error = function(e) e)
    if("error"  %in% class(error)) {
      rm(error)
      error = tryCatch({
        time = system.time({
          model = gllvm::gllvm(y = train_Y, X = data.frame(train_X),formula = ~X1+X2+X3+X4+X5, family = binomial("probit"), num.lv = dict[[as.character(tmp$species)]], starting.val = "zero", seed = 42)
        })},error = function(e) e)
    }
    if("error"  %in% class(error)) {
      rm(error)
      error = tryCatch({
        time = system.time({
          model = gllvm::gllvm(y = train_Y, X = data.frame(train_X),formula = ~X1+X2+X3+X4+X5, family = binomial("probit"), num.lv = dict[[as.character(tmp$species)]], starting.val = "random", seed = 42)
        })},error = function(e) e)
    }
    try({
      coefs = rbind(model$params$beta0, t(coef(model)$Xcoef))
      true_species_weights = rbind(rep(0.0, ncol(train_Y)), sim$species_weights)
      
      result_corr_acc[i,j] =  sim$corr_acc(gllvm::getResidualCov(model)$cov)
      result_env[i,j] = mean(as.vector(t(coef(model)$Xcoef) > 0) == as.vector(sim$species_weights > 0))
      result_rmse_env[i,j] =  sqrt(mean((as.vector(coefs) - as.vector(true_species_weights))^2))
      result_time[i,j] = time[3]
      pred = predict.gllvm(model, newX = data.frame(test_X), type = "response")
      sub_auc[[j]] = list(pred = pred, true = test_Y)
      post[[j]] = list(correlation=gllvm::getResidualCov(model)$cov)
      rm(model)
      gc()
      .torch$cuda$empty_cache()
      },silent = TRUE)
  }
  auc[[i]] = sub_auc
  diagnosis[[i]] = post
  
  gllvm = list(
    setup = setup[i,],
    result_corr_acc = result_corr_acc,
    result_env = result_env,
    result_rmse_env = result_rmse_env,
    result_time= result_time,
    auc = auc,
    post = diagnosis
  )
  saveRDS(gllvm, "results/1_gllvm_full.RDS")
}

