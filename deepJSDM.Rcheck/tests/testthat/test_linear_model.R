context("linear model")

source("utils.R")

sim = simulate_SDM()
X = sim$env_weights
Y = sim$response

test_succeeds("fit linear model", {
  model = createModel(X, Y)
  model = layer_dense(model, ncol(Y))
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
})


test_succeeds("predict with linear model", {
  model = createModel(X, Y)
  model = layer_dense(model, ncol(Y))
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
  predict(model, X)
})
