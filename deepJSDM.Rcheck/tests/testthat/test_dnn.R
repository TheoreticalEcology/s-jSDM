context("dnn")

source("utils.R")

sim = simulate_SDM()
X = sim$env_weights
Y = sim$response

test_succeeds("fit dnn", {
  model = createModel(X, Y)
  model = layer_dense(model, 10L, FALSE, FALSE)
  model = layer_dense(model, ncol(Y), FALSE, FALSE)
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
})


test_succeeds("predict dnn", {
  model = createModel(X, Y)
  model = layer_dense(model, 10L, FALSE, FALSE)
  model = layer_dense(model, ncol(Y), FALSE, FALSE)
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
  predict(model, X)
})



test_succeeds("dnn relu", {
  model = createModel(X, Y)
  model = layer_dense(model, 10L, "relu", FALSE)
  model = layer_dense(model, ncol(Y), FALSE, FALSE)
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
  predict(model, X)
})



test_succeeds("dnn bias", {
  model = createModel(X, Y)
  model = layer_dense(model, 10L, "relu", TRUE)
  model = layer_dense(model, ncol(Y), FALSE, FALSE)
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
  predict(model, X)
})



test_succeeds("dnn l1", {
  model = createModel(X, Y)
  model = layer_dense(model, 10L, "relu", FALSE, l1 = 0.1)
  model = layer_dense(model, ncol(Y), FALSE, FALSE)
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
  predict(model, X)
})



test_succeeds("dnn l2", {
  model = createModel(X, Y)
  model = layer_dense(model, 10L, "relu", FALSE, l2 = 0.1)
  model = layer_dense(model, ncol(Y), FALSE, FALSE)
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
  predict(model, X)
})



test_succeeds("dnn l2", {
  model = createModel(X, Y)
  model = layer_dense(model, 10L, "relu", FALSE, l2 = 0.1)
  model = layer_dense(model, ncol(Y), FALSE, FALSE)
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
  predict(model, X)
})




test_succeeds("dnn l1 l2", {
  model = createModel(X, Y)
  model = layer_dense(model, 10L, "relu", FALSE, l1 = 0.1,l2 = 0.1)
  model = layer_dense(model, ncol(Y), FALSE, FALSE)
  model = compileModel(model)
  model = deepJ(model, epochs = 1L)
  predict(model, X)
})
