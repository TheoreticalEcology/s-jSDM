\donttest{
  
# # Basic workflow:
# ## simulate community:
# com = simulate_SDM(env = 3L, species = 5L, sites = 100L)
# 
# ## fit model:
# model = sLVM(Y = com$response,X = com$env_weights, 
#              iter = 50L, family = binomial("probit"))
# coef(model)
# summary(model)
# getCov(model)
# 
# # get latent variables and factors
# getLF(model)
# getLV(model)
# 
# ## get credible intervals:
# getCI(model)
# 
# ## fit model with interactions:
# model = sLVM(Y = com$response,
#              X = com$env_weights, formula = ~X1:X2 + X3, 
#              family = binomial("probit"))
# summary(model)
# 
# ## without intercept:
# 
# ## predict with model:
# preds = predict(model, newdata = com$env_weights)
# 
# ## predictions over parameter distribution:
# preds = predict(model, newdata = com$env_weights, mean_field = FALSE)
# dim(preds)
}
