\donttest{
# simulate sparse community:
com = simulate_SDM(env = 5L, species = 25L, sites = 100L, sparse = 0.5)

# tune regularization:
tune_results = sjSDM_cv(X = com$env_weights, 
                        Y = com$response,
                        tune = "random", # random steps in tune-paramter space
                        CV = 3L, # 3-fold cross validation
                        tune_steps = 25L,
                        alpha_cov = seq(0, 1, 0.1),
                        alpha_coef = seq(0, 1, 0.1),
                        lambda_cov = seq(0, 0.1, 0.001), 
                        lambda_coef = seq(0, 0.1, 0.001),
                        n_cores = 4L, # small models can be also run in parallel on the GPU
                        iter = 2L # we can pass arguments to sjSDM via ...
                        )

# print overall results:
tune_results

# summary (mean values over CV for each tuning step)
summary(tune_results)

# visualize tuning and best points:
best = plot(tune_results, perf = "AUC")

# fit model with new regularization paramter:
model = sjSDM(X = com$env_weights,
              Y = com$response,
              l1_coefs = best[["l1_coef"]],
              l2_coefs = best[["l2_coef"]],
              l1_cov = best[["l1_cov"]],
              l2_cov = best[["l2_cov"]])

summary(model)
}
