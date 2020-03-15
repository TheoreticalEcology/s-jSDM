\donttest{
# simulate community:
com = simulate_SDM(env = 3L, species = 5L, sites = 100L)

# fit model:
model = sjSDM(X = com$env_weights, Y = com$response, iter = 10L)
coef(model)
summary(model)
getCov(model)

# calculate post-hoc p-values:
p = getSe(model)
summary(p)

# or turn the option in the sjSDM function on:
model = sjSDM(X = com$env_weights, Y = com$response, se = TRUE)
summary(model)

# fit model with interactions:
model = sjSDM(X = com$env_weights, Y = com$response, formula = ~X1:X2 + X3, se = TRUE)
summary(model)

# without intercept:
model = sjSDM(X = com$env_weights, Y = com$response, formula = ~ 0 + X1:X2 + X3, se = TRUE)
summary(model)

# predict with model:
preds = predict(model, newdata = com$env_weights)
}
