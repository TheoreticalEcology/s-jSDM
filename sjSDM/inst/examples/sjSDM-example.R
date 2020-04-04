\donttest{
  
# Basic workflow:
## simulate community:
com = simulate_SDM(env = 3L, species = 5L, sites = 100L)

## fit model:
model = sjSDM(Y = com$response,env = com$env_weights, iter = 10L)
coef(model)
summary(model)
getCov(model)

## calculate post-hoc p-values:
p = getSe(model)
summary(p)

## or turn on the option in the sjSDM function:
model = sjSDM(Y = com$response, env = com$env_weights, se = TRUE)
summary(model)

## fit model with interactions:
model = sjSDM(Y = com$response,
              env = envLinear(data = com$env_weights, formula = ~X1:X2 + X3), se = TRUE)
summary(model)

## without intercept:
model = sjSDM(Y = com$response,
              env = envLinear(data = com$env_weights, formula = ~0+X1:X2 + X3), se = TRUE)
summary(model)

## predict with model:
preds = predict(model, newdata = com$env_weights)


# With random intercepts on site:
RR = sample(1:5, 100, TRUE) 
Re = rnorm(5, 0, 0.1) # simulate random intercepts (5 sites)
SiteRE = Re[RR]
com = simulate_SDM(env = 3L, species = 5L, sites = 100L, Re = SiteRE)
model = sjSDM(Y = com$response,
              spatial = spatialRE(as.factor(RR)), # provide factors/sites
              env = envLinear(data = com$env_weights, formula = ~.), se = TRUE)
summary(model)
ranef(model)
cor(ranef(model), Re)



# Regularization
## lambda is the regularization strength
## alpha weights the lasso or ridge penalty:
## - alpha = 0 --> pure lasso
## - alpha = 1.0 --> pure ridge
model = sjSDM(Y = com$response, 
              # mix of lasso and ridge
              env = envLinear(com$env_weights, lambda = 0.01, alpha = 0.5), 
              # we can do the same for the species-species associations
              biotic = bioticStruct(lambda = 0.01, alpha = 0.5)
              )
summary(model)
coef(model)
getCov(model)



# Deep neural network
## we can fit also a deep neural network instead of a linear model:
model = sjSDM(Y = com$response,
              env = envDNN(com$env_weights, hidden = c(10L, 10L, 10L)))
summary(model)
getCov(model)
plot(model)
pred = predict(model, newdata = com$env_weights)

## extract weights
weights = getWeights(model)

## we can also assign weights:
setWeights(model, weights)

## with regularization:
model = sjSDM(Y = com$response, 
              # mix of lasso and ridge
              env = envDNN(com$env_weights, lambda = 0.01, alpha = 0.5), 
              # we can do the same for the species-species associations
              biotic = bioticStruct(lambda = 0.01, alpha = 0.5)
              )
getCov(model)
getWeights(model)

}
