context("sjSDM_model")

source("utils.R")

testthat::test_that("sjSDM functionality", {
  skip_if_no_torch()
  
  com = simulate_SDM(env = 3L, species = 5L, sites = 100L)
  
  testthat::expect_error({
    X = com$env_weights
    Y = com$response
    model = sjSDM_model(input_shape = 3L)
    model %>% 
      layer_dense(units = 10L, activation = "tanh") %>% 
      layer_dense(units = 10L, activation = "relu") %>% 
      layer_dense(units = 5L) 
    model %>% 
      compile(df = 50L, 
              optimizer = optimizer_adamax(learning_rate = 0.01), 
              l1_cov = 0.0001, 
              l2_cov = 0.0001)
    model %>% 
      summary
    model %>% 
      fit(X = X, Y = Y, epochs = 1L, batch_size = 10L)
    plot(model)
    w = getWeights(model)
    setWeights(model, w)
    getCov(model)
    p = predict(model, X)
  }, NA)
  
  
  
  testthat::expect_error({
    X = com$env_weights
    Y = com$response
    model = sjSDM_model(input_shape = 3L)
    model %>% 
      layer_dense(units = 3L) %>% 
      layer_dense(units = 5L) %>% 
      layer_dense(units = 5L) 
    model %>% 
      compile(df = 2L, 
              optimizer = optimizer_adamax(learning_rate = 0.01), 
              l1_cov = 0.0001, 
              l2_cov = 0.0001)
    model %>% 
      summary
    model %>% 
      fit(X = X, Y = Y, epochs = 1L, batch_size = 10L)
    plot(model)
    w = getWeights(model)
    setWeights(model, w)
    getCov(model)
    p = predict(model, X)
  }, NA)
  
  com = simulate_SDM(env = 7L, species = 25L, sites = 100L)
  
  testthat::expect_error({
    X = com$env_weights
    Y = com$response
    model = sjSDM_model(input_shape = 7L)
    model %>% 
      layer_dense(units = 3L) %>% 
      layer_dense(units = 5L) %>% 
      layer_dense(units = 25L) 
    model %>% 
      compile(df = 2L, 
              optimizer = optimizer_adamax(learning_rate = 0.01), 
              l1_cov = 0.0001, 
              l2_cov = 0.0001)
    model %>% 
      summary
    model %>% 
      fit(X = X, Y = Y, epochs = 1L, batch_size = 10L)
    plot(model)
    w = getWeights(model)
    setWeights(model, w)
    getCov(model)
    p = predict(model, X)
  }, NA)
  
  com = simulate_SDM(env = 7L, species = 25L, sites = 100L)
  
  testthat::expect_error({
    X = com$env_weights
    Y = com$response
    model = sjSDM_model(input_shape = 7L)
    model %>% 
      layer_dense(units = 11L) %>% 
      layer_dense(units = 5L) %>% 
      layer_dense(units = 25L) 
    model %>% 
      compile(df = 25L, 
              optimizer = optimizer_adamax(learning_rate = 0.01), 
              l1_cov = 0.0001, 
              l2_cov = 0.0001)
    model %>% 
      summary
    model %>% 
      fit(X = X, Y = Y, epochs = 3L, batch_size = 20L)
    plot(model)
    w = getWeights(model)
    setWeights(model, w)
    getCov(model)
    p = predict(model, X, batch_size = 10L)
  }, NA)
  
  testthat::expect_error({
    weights = getWeights(model)
    model2 = sjSDM_model(input_shape = 7L)
    model2 %>% 
      layer_dense(units = 11L) %>% 
      layer_dense(units = 5L) %>% 
      layer_dense(units = 25L) 
    model2 %>% 
      compile(df = 25L, 
              optimizer = optimizer_adamax(learning_rate = 0.01), 
              l1_cov = 0.0001, 
              l2_cov = 0.0001)
    setWeights(model2, weights)
    if(!any(as.vector(model2$weights_numpy[[1]][[1]] == weights$layers[[1]][[1]]))) stop("Error...")
  }, NA)
  
})