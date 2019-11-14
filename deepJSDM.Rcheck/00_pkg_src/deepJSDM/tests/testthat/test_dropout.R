context("dropout")

source("utils.R")

sim = simulate_SDM()
X = sim$env_weights
Y = sim$response

test_succeeds("fit dnn droput", {
  model = createModel(X, Y)
  model = layer_dense(model, 30L, FALSE, FALSE)
  model = layer_dropout(model, 0.3)
  model = layer_dense(model, ncol(Y), FALSE, FALSE)
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
})


test_succeeds("predict dnn dropout", {
  model = createModel(X, Y)
  model = layer_dense(model, 30L, FALSE, FALSE)
  model = layer_dropout(model, 0.3)
  model = layer_dense(model, ncol(Y), FALSE, FALSE)
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
  predict(model, X)
})



test_succeeds("predict dnn dropout with rate", {
  model = createModel(X, Y)
  model = layer_dense(model, 30L, FALSE, FALSE)
  model = layer_dropout(model, 0.3)
  model = layer_dense(model, ncol(Y), FALSE, FALSE)
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
  predict(model, X, train = FALSE)
})
