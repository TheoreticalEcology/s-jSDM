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
#  model = sjSDM(Y = com$response, env = linear(data = com$env_weights, formula = ~ X1*X2,), iter = 50L, se = TRUE)
#  summary(model)
#  

## ----eval=FALSE----------------------------------------------------------
#  model = sjSDM(Y = com$response, env = linear(data = com$env_weights, formula = ~0+ I(X1^2)), iter = 50L, se = TRUE)
#  summary(model)

## ----eval=FALSE----------------------------------------------------------
#  model = sjSDM(Y = com$response, env = linear(data = com$env_weights, formula = ~0+ I(X1^2),lambda = 0.5), iter = 50L)
#  summary(model)

## ----eval=FALSE----------------------------------------------------------
#  model = sjSDM(Y = com$response,
#                env = linear(data = com$env_weights, formula = ~0+ I(X1^2),lambda = 0.5),
#                biotic = bioticStruct(lambda =0.1),
#                iter = 50L)
#  summary(model)

## ----eval=FALSE----------------------------------------------------------
#  model = sjSDM(Y = com$response,
#                env = linear(data = com$env_weights, formula = ~0+ I(X1^2),lambda = 0.5),
#                biotic = bioticStruct(lambda =0.1, on_diag = FALSE,inverse = TRUE),
#                iter = 50L)
#  summary(model)

## ----eval=FALSE----------------------------------------------------------
#  com = simulate_SDM(env = 3L, species = 5L, sites = 100L)
#  X = com$env_weights
#  Y = com$response
#  
#  # three fully connected layers with relu as activation function
#  model = sjSDM(Y = Y,
#                env = DNN(data = X, formula = ~., hidden = c(10L, 10L, 10L), activation = "relu"),
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
#  XYcoords = matrix(rnorm(200), 100, 2)
#  
#  model = sjSDM(Y, env = linear(X, ~X1+X2), spatial = linear(XYcoords, ~0+X1:X2))
#  summary(model)

## ----eval=FALSE----------------------------------------------------------
#  
#  model = sjSDM(Y, env = linear(X, ~X1+X2), spatial = linear(XYcoords, ~0+X1:X2, lambda = 0.4))
#  summary(model)

## ----eval=FALSE----------------------------------------------------------
#  
#  model = sjSDM(Y, env = linear(X, ~X1+X2), spatial = DNN(XYcoords, ~0+X1:X2, lambda = 0.4))
#  summary(model)

## ----eval=FALSE----------------------------------------------------------
#  
#  model = sjSDM(Y, env = DNN(X, ~X1+X2), spatial = DNN(XYcoords, ~0+X1:X2, lambda = 0.4))
#  summary(model)

