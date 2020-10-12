library(sjSDM)
torch$cuda$manual_seed(42L)
torch$manual_seed(42L)

load("data/eDNA/SDM_data.RData")
tune = readRDS("results/tuning_fungi_v2.RDS")

occ = ttab_ds[[7]]
dim(occ)
rates = apply(occ, 2, mean)
occ = occ[,rates > 2/125]
dim(occ)

env_scaled = scale(env2)


best1 = tune[order(tune$logLik),][1,]
best2 = tune[order(tune$logLik),][2,]

model1 = sjSDM(Y = occ,
              env = linear(env_scaled, 
                           lambda = best1[["lambda_coef"]],
                           alpha = best1[["alpha_coef"]]),
              biotic = bioticStruct(lambda = best1[["lambda_cov"]],
                                    alpha = best1[["alpha_cov"]],
                                    df = dim(occ)[2]),
              learning_rate = 0.001, iter = 150, step_size = 5L, device = 2, link = "logit", sampling = 1000L
)

model2= sjSDM(Y = occ,
               env = linear(env_scaled, 
                            lambda = best2[["lambda_coef"]],
                            alpha = best2[["alpha_coef"]]),
               biotic = bioticStruct(lambda = best2[["lambda_cov"]],
                                     alpha = best2[["alpha_cov"]],
                                     df = dim(occ)[2]),
               learning_rate = 0.001, iter = 150L, step_size = 5L, device = 2, link = "logit",  sampling = 1000L
)
base = sjSDM(Y = occ,
             env = linear(env_scaled),
             biotic = bioticStruct(df = dim(occ)[2]),
             learning_rate = 0.001, iter = 150L, step_size = 5L, device = 2, link = "logit",  sampling = 1000L
)

saveRDS(list(base = base, best1 = model1, best2 = model2), file = "results/5_fungi_models.RDS")
