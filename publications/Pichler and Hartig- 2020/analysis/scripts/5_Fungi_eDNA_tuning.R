library(sjSDM)
torch$cuda$manual_seed(42L)
torch$manual_seed(42L)

load("data/eDNA/SDM_data.RData")

occ = ttab_ds[[7]]
dim(occ)
rates = apply(occ, 2, mean)
occ = occ[,rates > 2/125]
dim(occ)

env_scaled = scale(env2)


tune = sjSDM_cv(env = env_scaled, Y = occ, 
                learning_rate = 0.001, iter = 150L, CV = nrow(occ),
                n_gpu = 3, n_cores = 3, tune_steps = 40,
                lambda_cov =  2.0^seq(-9, -2, length.out = 10),
                lambda_coef = 0.1,
                alpha_cov = seq(0, 1, 0.05),
                alpha_coef = 0.5,
                sampling = 1000L,
                biotic =  bioticStruct(df=dim(occ)[2]),
                step_size = 5L,
                link = "logit",
                blocks = 3L
                )



saveRDS(tune$short_summary, file = "results/tuning_fungi_v2.RDS")
