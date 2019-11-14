context("optimizer")

source("utils.R")

sim = simulate_SDM()
X = sim$env_weights
Y = sim$response

test_succeeds("adamax", {
  model = createModel(X, Y)
  model = layer_dense(model, ncol(Y))
  model = compileModel(model, optimizer = "adamax")
  model = deepJ(model, epochs = 1L)
})



test_succeeds("LBFGS", {
  model = createModel(X, Y)
  model = layer_dense(model, ncol(Y))
  model = compileModel(model, optimizer = "LBFGS")
  model = deepJ(model, epochs = 1L)
})



test_succeeds("adamax parameter", {
  model = createModel(X, Y)
  model = layer_dense(model, ncol(Y))
  model = compileModel(model, optimizer = "adamax", control = list(weight_decay = 1L))
  model = deepJ(model, epochs = 1L)
})



test_succeeds("LBFGS parameter", {
  model = createModel(X, Y)
  model = layer_dense(model, ncol(Y))
  model = compileModel(model, optimizer = "LBFGS", control = list(max_eval = 10L))
  model = deepJ(model, epochs = 1L)
})
