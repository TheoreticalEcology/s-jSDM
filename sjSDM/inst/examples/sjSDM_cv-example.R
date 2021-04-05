\donttest{
# simulate sparse community:
com = simulate_SDM(env = 5L, species = 25L, sites = 100L, sparse = 0.5)

# tune regularization:
tune_results = sjSDM_cv(Y = com$response,
                        env = com$env_weights, 
                        tune = "random", # random steps in tune-paramter space
                        CV = 3L, # 3-fold cross validation
                        tune_steps = 25L,
                        alpha_cov = seq(0, 1, 0.1),
                        alpha_coef = seq(0, 1, 0.1),
                        lambda_cov = seq(0, 0.1, 0.001), 
                        lambda_coef = seq(0, 0.1, 0.001),
                        n_cores = 2L, 
                        # small models can be also run in parallel on the GPU
                        iter = 2L # we can pass arguments to sjSDM via... 
                        )

# print overall results:
tune_results

# summary (mean values over CV for each tuning step)
summary(tune_results)

# visualize tuning and best points:
best = plot(tune_results, perf = "logLik")

# fit model with new regularization paramter:
model = sjSDM(Y = com$response,
              env = linear(com$env_weights, 
                              lambda = best[["lambda_coef"]],
                              alpha = best[["alpha_coef"]]),
              biotic = bioticStruct(lambda = best[["lambda_cov"]],
                                    alpha = best[["alpha_cov"]]),
              iter = 2L # increase it for your own data 
              )

summary(model)
}
