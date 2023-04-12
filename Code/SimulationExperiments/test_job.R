library(sjSDM)
?sjSDM
# Basic workflow:
## simulate community:
com = simulate_SDM(env = 3L, species = 7L, sites = 100L)

form = "~X1+X1:X2"
## fit model:
model = sjSDM(Y = com$response,env = linear(com$env_weights, as.formula(form)), iter = 50L) 
# increase iter for your own data 