\donttest{
com = simulate_SDM(env = 3L, species = 5L, sites = 100L)
X = com$env_weights
Y = com$response

model = sjSDM_DNN(X = com$env_weights, Y = com$response, hidden = c(5L, 5L, 5L), 
                  activation = c("relu", "tanh", "relu"), iter = 10L)

# fit with interaction
model = sjSDM_DNN(X = com$env_weights, Y = com$response, formula = ~0+X1:X2+X3 ,
                  hidden = c(5L, 5L, 5L), 
                  activation = c("relu", "tanh", "relu"), iter = 10L)


# predict on fitted data
preds = predict(model)

# predict on new data
preds = predict(model, newdata = X)

# get species associations
sp = getCov(model)

# extract weights
weights = getWeights(model)

# we can also assign weights:
setWeights(model, weights)
}