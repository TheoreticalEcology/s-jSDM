context("variational dense")

source("utils.R")

sim = simulate_SDM()
X = sim$env_weights
Y = sim$response

test_succeeds("fit dnn vb", {
  model = createModel(X, Y)
  model = layer_varational_dense(model, 10L, 1.0, 0.1)
  model = layer_dense(model, ncol(Y), FALSE, FALSE)
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
})


test_succeeds("predict dnn vb", {
  model = createModel(X, Y)
  model = layer_varational_dense(model, 10L, 1.0, 0.1)
  model = layer_dense(model, ncol(Y), FALSE, FALSE)
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
  predict(model, X)
})



test_succeeds("dnn relu vb", {
  model = createModel(X, Y)
  model = layer_varational_dense(model, 10L,activation = "relu", 1.0, 0.1)
  model = layer_dense(model, ncol(Y), FALSE, FALSE)
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
  predict(model, X)
})
