library(sjSDM)
library(gllvm)
sjSDM:::.onLoad()

create = function(env = 5L, n = 100L, sp = 50L, l = 5L) {
  E = matrix(runif(env*n,-1,1), n, env) # environment
  SPW = matrix(rnorm(sp*env), env, sp) # species weights
  
  L = matrix(rnorm(l*n), n, l) # latent variables
  SPL = matrix(rnorm(l*sp), l, sp) # Factor loadings
  
  Y = E %*% SPW + L %*% SPL
  Occ = ifelse(Y > 0, 1, 0)
  
  sigma =  t(SPL) %*% SPL
  
  corr_acc = function(cor) {
    ind = lower.tri(sigma)
    true = sigma[ind]
    pred = cor[ind]
    d = sum((true < 0) == (pred < 0))
    return(d/sum(lower.tri(sigma)))
  }
  return(list(Y=Occ, X = E, L = L, SPL = SPL, SPW = SPW, sigma =  sigma, corr_acc = corr_acc))
}
data_10 = lapply(1:5, function(l) lapply(1:5, function(s) create(env = 3L, n = 300L, sp = 10L, l = l)))
data_50 = lapply(1:5, function(l) lapply(1:5, function(s) create(env = 3L, n = 300L, sp = 50L, l = l)))
data_100 = lapply(1:5, function(l) lapply(1:5, function(s) create(env = 3L, n = 300L, sp = 100L, l = l)))
data = list(d10 = data_10, d50 = data_50, d100 = data_100)
sjSDM_cov =  gllvm_cov =sjSDM_rmse  = gllvm_rmse = matrix(NA, 5, 5)


sjSDM_res = vector("list", 3)
for(d in 1:3) {
  sjSDM_cov = sjSDM_rmse = sjSDM_time =  matrix(NA, 5, 5)
  for(i in 1:5) {
    for(j in 1:5) {
      m = fa$Model_LVM()
      time = system.time({
        m$fit(X = data[[d]][[i]][[j]]$X, Y =  data[[d]][[i]][[j]]$Y, df = as.integer(i), guide = "DiagonalNormal", lr = list(0.1), batch_size = 30L, epochs = 100L, scale_mu = 1.0)
      })
      sp_sjSDM = m$covariance
      sjSDM_cov[i,j] = data[[d]][[i]][[j]]$corr_acc(sp_sjSDM)
      sjSDM_rmse[i,j] = sqrt(mean(as.vector(m$posterior_samples$mu$squeeze()$data$cpu()$mean(dim=0L)$numpy() - data[[d]][[i]][[j]]$SPW)^2))
      sjSDM_time[i,j] = time[3]
    }
  }
  sjSDM_res[[d]] = list(cov = sjSDM_cov, rmse = sjSDM_rmse, time = sjSDM_time)
}

saveRDS(sjSDM_res, "Code/Results/sjSDM.RDS")

gllvm_res = vector("list", 3)
for(d in 1:3) {
  gllvm_cov = gllvm_rmse = gllvm_time =  matrix(NA, 5, 5)
  for(i in 1:5) {
    for(j in 1:5) {
      try({
        time = system.time({
          m = gllvm::gllvm(y = data[[d]][[i]][[j]]$Y, X = data.frame(data[[d]][[i]][[j]]$X),formula = ~0+., family = binomial("logit"), num.lv = i, seed = 42)
        })
        sp_sjSDM = getResidualCor(m)
        gllvm_cov[i,j] = data[[d]][[i]][[j]]$corr_acc(sp_sjSDM)
        gllvm_rmse[i,j] = sqrt(mean(as.vector(t(coef(m)$Xcoef)  - data[[d]][[i]][[j]]$SPW)^2))
        gllvm_time[i,j] = time[3]
      }, silent = TRUE)
    }
  }
  gllvm_res[[d]] = list(cov = gllvm_cov, rmse = gllvm_rmse, time = gllvm_time)
}

saveRDS(gllvm_res, "Code/Results/gllvm.RDS")