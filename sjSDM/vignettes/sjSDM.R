## ---- echo = F, message = F----------------------------------------------
set.seed(123)

## ----global_options, include=FALSE---------------------------------------
knitr::opts_chunk$set(fig.width=7, fig.height=4.5, fig.align='center', warning=FALSE, message=FALSE, cache = F)

## ----eval=FALSE----------------------------------------------------------
#  library(sjSDM)

## ---- eval = F-----------------------------------------------------------
#  install_sjSDM()

## ----eval = F------------------------------------------------------------
#  vignette("Dependencies", package = "sjSDM")

## ----eval=FALSE----------------------------------------------------------
#  citation("sjSDM")

## ----eval=FALSE----------------------------------------------------------
#  com = simulate_SDM(env = 3L, species = 5L, sites = 100L)

## ----eval=FALSE----------------------------------------------------------
#  model = sjSDM(Y = com$response, env = com$env_weights, iter = 10L)

## ----eval=FALSE----------------------------------------------------------
#  coef(model)
#  summary(model)
#  getCov(model)

## ----eval=FALSE----------------------------------------------------------
#  model = sjSDM(Y = com$response, env = envLinear(data = com$env_weights, formula = ~ X1*X2,), iter = 50L, se = TRUE)
#  summary(model)
#  

## ----eval=FALSE----------------------------------------------------------
#  model = sjSDM(Y = com$response, env = envLinear(data = com$env_weights, formula = ~0+ I(X1^2),), iter = 50L, se = TRUE)
#  summary(model)

## ----eval=FALSE----------------------------------------------------------
#  model = sjSDM(Y = com$response,
#                env = envLinear(data = com$env_weights, formula = ~.,),
#                spatial = spatialRE(re = as.factor(1:100)),
#                iter = 50L, se = TRUE)
#  summary(model)
#  ranef(model)
#  

## ----eval=FALSE----------------------------------------------------------
#  com = simulate_SDM(env = 3L, species = 5L, sites = 100L)
#  X = com$env_weights
#  Y = com$response
#  
#  # three fully connected layers with relu as activation function
#  model = sjSDM(Y = Y,
#                env = envDNN(data = X, formula = ~., hidden = c(10L, 10L, 10L), activation = "relu"),
#                iter = 50L, se = TRUE)
#  summary(model)

## ----eval=FALSE----------------------------------------------------------
#  getCov(model) # species association matrix
#  pred = predict(model) # predict on fitted data
#  pred = predict(model, newdata = X) # predict on new data

## ----eval=FALSE----------------------------------------------------------
#  weights = getWeights(model) # get layer weights and sigma
#  setWeights(model, weights)

## ----eval=FALSE----------------------------------------------------------
#  plot(model)

## ----eval=FALSE----------------------------------------------------------
#  com = simulate_SDM(env = 3L, species = 5L, sites = 100L)
#  X = com$env_weights
#  Y = com$response
#  
#  model = sjSDM_model(input_shape = 3L)
#  model %>%
#    layer_dense(units = 10L, activation = "tanh") %>%
#    layer_dense(units = 10L, activation = "relu") %>%
#    layer_dense(units = 5L)
#  
#  model %>%
#    compile(df = 50L, optimizer = optimizer_adamax(learning_rate = 0.01), l1_cov = 0.0001, l2_cov = 0.0001)
#  
#  model %>%
#    fit(X = X, Y = Y, epochs = 10L, batch_size = 10L)
#  
#  summary(model)
#  
#  getCov(model)
#  
#  weights = getWeights(model)
#  
#  setWeights(model, weights)
#  
#  pred = predict(model, X)
#  
#  plot(model)

