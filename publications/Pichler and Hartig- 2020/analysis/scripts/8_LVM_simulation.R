library(sjSDM)
library(Hmsc)
library(gllvm)
library(snow)
torch$cuda$manual_seed(42L)
set.seed(42)

create = function(env = 5L, n = 100L, sp = 50L, l = 5L, SPW_range = c(-1, 1), SPL_range = c(-1, 1)) {
  E = matrix(runif(env*n,-1,1), n, env) # environment
  SPW = matrix(runif(sp*env, SPW_range[1], SPW_range[2]), env, sp) # species weights
  
  L = matrix(rnorm(l*n), n, l) # latent variables
  SPL = matrix(runif(l*sp, SPL_range[1], SPL_range[2]), l, sp) # Factor loadings
  
  Y = E %*% SPW + L %*% SPL
  Occ = apply(Y, 1:2, function(p) rbinom(1, 1, pnorm(p)))
  
  sigma =  t(SPL) %*% SPL + diag(1.0, sp)
  
  if(ncol(E) == 0) E = matrix(1.0, n, 1)
  
  corr_acc = function(cor) {
    ind = lower.tri(sigma)
    true = sigma[ind]
    pred = cor[ind]
    d = sum((true < 0) == (pred < 0))
    return(d/sum(lower.tri(sigma)))
  }
  return(list(Y=Occ, X = E, L = L, SPL = SPL, SPW = SPW, sigma =  sigma, corr_acc = corr_acc))
}
rmse = function(true, obs) sqrt(mean(( true - obs) ^2))


data = list(
  data_10_E =     lapply(1:5, function(l) list(create(env = 2L, n = 200L, sp = 10L, l = 0L  ))),
  data_10_EL =    lapply(1:5, function(l) lapply(1:5, function(s) create(env = 2L, n = 200L, sp = 10L, l = l, SPL_range = c(-1, 1) ))),
  data_10_EL_51 = lapply(1:5, function(l) lapply(1:5, function(s) create(env = 2L, n = 200L, sp = 10L, l = l, SPL_range = c(-1, 1)/5  ))),
  data_10_EL_15 = lapply(1:5, function(l) lapply(1:5, function(s) create(env = 2L, n = 200L, sp = 10L, l = l, SPL_range = c(-1, 1)*5   ))),
  
  data_50_E =     lapply(1:5, function(l) list(create(env = 2L, n = 200L, sp = 50L, l = 0L  ))),
  data_50_EL =    lapply(1:5, function(l) lapply(1:5, function(s) create(env = 2L, n = 200L, sp = 50L, l = l, SPL_range = c(-1, 1) ))),
  data_50_EL_51 = lapply(1:5, function(l) lapply(1:5, function(s) create(env = 2L, n = 200L, sp = 50L, l = l, SPL_range = c(-1, 1)/5  ))),
  data_50_EL_15 = lapply(1:5, function(l) lapply(1:5, function(s) create(env = 2L, n = 200L, sp = 50L, l = l, SPL_range = c(-1, 1)*5   ))),
  
  data_100_E =     lapply(1:5, function(l) list(create(env = 2L, n = 200L, sp = 100L, l = 0L  ))),
  data_100_EL =    lapply(1:5, function(l) lapply(1:5, function(s) create(env = 2L, n = 200L, sp = 100L, l = l, SPL_range = c(-1, 1) ))),
  data_100_EL_51 = lapply(1:5, function(l) lapply(1:5, function(s) create(env = 2L, n = 200L, sp = 100L, l = l, SPL_range = c(-1, 1)/5  ))),
  data_100_EL_15 = lapply(1:5, function(l) lapply(1:5, function(s) create(env = 2L, n = 200L, sp = 100L, l = l, SPL_range = c(-1, 1)*5   )))
)


## sjSDM, Hmsc, and GLLVM
sjSDM_res = Hmsc_res = gllvm_res = list()

cl = snow::makeCluster(6L)
snow::clusterExport(cl, list("data", "rmse"), envir = environment())
ev = snow::clusterEvalQ(cl, {
  library(Hmsc)
  library(gllvm)
  library(sjSDM)
  set.seed(42)
  })


sjSDM_res = 
  snow::parLapply(cl, 1:length(data), function(d) {
    sjSDM_cov = sjSDM_rmse = sjSDM_cov_rmse = matrix(NA, 5, 5)
    for(i in 1:5) {
      for(j in 1:length(data[[d]][[i]])) {
      sjSDM = sjSDM(data[[d]][[i]][[j]]$Y, env = linear(data[[d]][[i]][[j]]$X, ~0+.), step_size = 10L, 
                    iter = 100L, device = 2, learning_rate = 0.005, family = binomial("probit"))
      sp_sjSDM = cov2cor(getCov(sjSDM))
      
      sjSDM_cov[i,j] = data[[d]][[i]][[j]]$corr_acc(sp_sjSDM)
      sjSDM_cov_rmse[i,j] = sqrt(mean((data[[d]][[i]][[j]]$sigma - sp_sjSDM)^2))
      sjSDM_rmse[i,j] = rmse(coef(sjSDM)[[1]], t(data[[d]][[i]][[j]]$SPW))
      }
    }
    return(list(cov = sjSDM_cov,cov_rmse = sjSDM_cov_rmse, rmse = sjSDM_rmse))
})

save(sjSDM_res, Hmsc_res, gllvm_res,  file = "results/LVMsimulation_scenarios.RData")

snow::stopCluster(cl)

cl = snow::makeCluster(12L)
snow::clusterExport(cl, list("data", "rmse"), envir = environment())
ev = snow::clusterEvalQ(cl, {
  library(Hmsc)
  library(gllvm)
  set.seed(42)
  })


gllvm_res = 
  snow::parLapply(cl, 1:length(data), function(d) {
    
   gllvm_cov = gllvm_cov_rmse = gllvm_rmse = matrix(NA, 5, 5) # for acc/rmse/acc/rmse/n_latent
   for(i in 1:5) {
    for(j in 1:length(data[[d]][[i]])){
        if(length(data[[d]][[i]]) == 1) nl = 1
        else nl = i 
        error = tryCatch({
        time = system.time({
        model = gllvm::gllvm(y = data[[d]][[i]][[j]]$Y, X = data.frame(data[[d]][[i]][[j]]$X), family = binomial("probit"), num.lv = nl, seed = 42)
        })},error = function(e) e)
        if("error"  %in% class(error)) {
          rm(error)
          error = tryCatch({
            time = system.time({
              model = gllvm::gllvm(y = data[[d]][[i]][[j]]$Y, X = data.frame(data[[d]][[i]][[j]]$X),  family = binomial("probit"), num.lv = nl, starting.val = "zero", seed = 42)
            })},error = function(e) e)
        }
        if("error"  %in% class(error)) {
          rm(error)
          error = tryCatch({
            time = system.time({
              model = gllvm::gllvm(y = data[[d]][[i]][[j]]$Y, X = data.frame(data[[d]][[i]][[j]]$X),  family = binomial("probit"), num.lv = nl, starting.val = "random", seed = 42)
            })},error = function(e) e)
        }
        try({
          cov = cov2cor(gllvm::getResidualCov(model)$cov)
           gllvm_cov[i,j] = data[[d]][[i]][[j]]$corr_acc(cov)
           gllvm_cov_rmse[i,j] = rmse(data[[d]][[i]][[j]]$sigma, cov)
           gllvm_rmse[i,j] = rmse( t(data[[d]][[i]][[j]]$SPW), coef(model)$Xcoef)
        })
        }
   }
   return(list(cov = gllvm_cov,cov_rmse = gllvm_cov_rmse, rmse = gllvm_rmse))
})

save(sjSDM_res, Hmsc_res, gllvm_res,  file = "results/LVMsimulation_scenarios.RData")

Hmsc_res = 
  snow::parLapply(cl, 1:12, function(d) {
    
   Hmsc_cov = Hmsc_cov_rmse = Hmsc_rmse = matrix(NA, 5, 5) # for acc/rmse/acc/rmse/n_latent
   for(i in 1:5) {
    for(j in 1:length(data[[d]][[i]])){
        hmsc = list()
        studyDesign = data.frame(sample = as.factor(1:nrow(data[[d]][[i]][[j]]$Y)))
        rL = HmscRandomLevel(units = studyDesign$sample)
        model = Hmsc(Y = data[[d]][[i]][[j]]$Y, XData = data.frame(data[[d]][[i]][[j]]$X), XFormula = ~0 + .,
                     studyDesign = studyDesign, ranLevels = list(sample = rL), distr = "probit")
        model = sampleMcmc(model, thin = 50, samples = 1000, transient = 5000,verbose = 5000,
                           nChains = 1L) # 50,000 iterations
        cov = computeAssociations(model)[[1]]$mean
        beta = Hmsc::getPostEstimate(model, "Beta")$mean
        Hmsc_cov[i,j] = data[[d]][[i]][[j]]$corr_acc(cov)
        Hmsc_cov_rmse[i,j] = rmse(data[[d]][[i]][[j]]$sigma, cov)
        Hmsc_rmse[i,j] = rmse( data[[d]][[i]][[j]]$SPW, beta )
      }
   }
   return(list(cov = Hmsc_cov,cov_rmse = Hmsc_cov_rmse, rmse = Hmsc_rmse))
})

save(sjSDM_res, Hmsc_res, gllvm_res,  file = "results/LVMsimulation_scenarios.RData")

