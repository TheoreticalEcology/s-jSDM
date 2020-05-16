library(sjSDM)
torch$cuda$manual_seed(42L)
torch$manual_seed(42L)

load("data/eDNA/SDM_data.RData")

occ = ttab_ds[[7]]
dim(occ)
rates = apply(occ, 2, mean)
occ = occ[,rates > 1/125]
dim(occ)

env_scaled = as.matrix(mlr::normalizeFeatures(env2))


tune = sjSDM_cv(env = env_scaled, Y = occ, 
                learning_rate = 0.001, iter = 200L, step_size = 8L, 
                n_gpu = 3, n_cores = 3, tune_steps = 60,
                lambda_cov = 1.5^seq(-10, 4, length.out = 30),
                alpha_cov = seq(0, 1, 0.05),
                alpha_coef = seq(0, 1, 0.05)
                )
best = tune$short_summary[order(tune$short_summary$logLik),][1,]
model = sjSDM(Y = occ,
              env = linear(env_scaled, 
                           lambda = best[["lambda_coef"]],
                           alpha = best[["alpha_coef"]]),
              biotic = bioticStruct(lambda = best[["lambda_cov"]],
                                    alpha = best[["alpha_cov"]]),
              learning_rate = 0.001, iter = 200L, step_size = 8L, device = 2
)

base = sjSDM(Y = occ,
             env = linear(env_scaled),
             learning_rate = 0.001, iter = 200L, step_size = 8L, device = 2
)
saveRDS(list(model = model, base = base, best = best, tune = tune), file = "results/tuning_fungi.RDS")



