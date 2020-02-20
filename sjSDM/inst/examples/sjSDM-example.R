\donttest{
# simulate community:
com = simulate_SDM(env = 3L, species = 5L, sites = 100L)

# fit model:
model = sjSDM(X = com$env_weights, Y = com$response, iter = 10L)
coef(model)
summary(model)
getCov(model)

# fit model with interactions:
model = sjSDM(X = com$env_weights, Y = com$response, formula = ~X1:X2 + X3, iter = 10L)
summary(model)

# without intercept:
model = sjSDM(X = com$env_weights, Y = com$response, formula = ~ 0 + X1:X2 + X3, iter = 10L)
summary(model)

# predict with model:
preds = predict(model, newdata = com$env_weights)
}
