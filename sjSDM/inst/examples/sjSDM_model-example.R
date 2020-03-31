\donttest{
com = simulate_SDM(env = 3L, species = 5L, sites = 100L)
X = com$env_weights
Y = com$response

# the activation function in the last layer must be linear (NULL)
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
  fit(X = X, Y = Y, epochs = 10L, batch_size = 10L)

# species association matrix:
sp = getCov(model)

# you can also continue training:
model %>% 
  fit(X = X, Y = Y, epochs = 10L, batch_size = 10L)

weights = getWeights(model)
}