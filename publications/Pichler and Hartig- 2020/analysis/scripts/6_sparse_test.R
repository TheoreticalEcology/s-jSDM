library(deepJSDM)
useGPU(2L)
set.seed(42L)

data = simulate_SDM(sites = 400, species = 200, sparse = 0.5)


folds = caret::createFolds(data$response[,1], k = 10L)
lambda = 0.5
alpha = 0.2
lambdas = seq(0.0001, 0.5, length.out = 20)

auc_overall = matrix(NA, 20, 10)
covs = vector("list", 10)
for(i in 1:20){
  tmp_auc = matrix(NA, 10, 2)
  tmp_cov = vector("list", 10L)
  for(j in 1:10){
    train_X = data$env_weights[-folds[[j]],]
    train_Y = data$response[-folds[[j]],]
    test_X = data$env_weights[folds[[j]],]
    test_Y = data$response[folds[[j]],]
    
    lambda = lambdas[i]
    model = createModel(train_X, train_Y)
    model = layer_dense(model, ncol(data$response),activation = FALSE, bias = FALSE)
    model = compileModel(model, nLatent = 400L,lr = 0.01, optimizer = "adamax", l2 = (1-alpha)*lambda, l1 = alpha*lambda)
    model = deepJ(model, epochs = 100L, batch_size = 40L)
    
    preds = predict(model, test_X)
    aucs = sapply(1:200, function(k) Metrics::auc(test_Y[,k], preds[,k]))
    tmp_auc[j,] = c(mean(aucs), sd(aucs))
    
    tmp_cov[[j]] = model$sigma()
    
    rm(model)
    gc()
    .torch$cuda$empty_cache()
    }
  auc_overall[i,] = tmp_auc[,1]
  covs[[i]] = tmp_cov
  cat("Mean AUC: ", mean(tmp_auc[,1]), "\n")
}

save(auc_overall, covs, file = "results/sparse_test.RData")
# data$corr_acc(model$sigma())


# true = data$correlation
# true = true[lower.tri(true)]
# 
# ss = (model$sigma())
# ss = ss[lower.tri(ss)]
# 
# 
# ss = ifelse(abs(ss) < 0.01, 0.0, ss)
# 
# data$corr_acc(ss)
# data$corr_acc(matrix(0, 200,200))
# 
# 
# library(gllvm)
# m = gllvm(data$response, data$env_weights, formula = ~., family = binomial("probit"))
# ss2 = gllvm::getResidualCov(m)$cov
# ss2 = ss2[lower.tri(ss2)]
# 
# 
# ss2 = ifelse(abs(ss2) < 0.000001, 0.0, ss2)
# 
# data$corr_acc(ss2)
# 
# data$corr_acc(gllvm::getResidualCov(m)$cov)
# 
# 
# 
# 
# save(env_scaled, occ_high, result, )
