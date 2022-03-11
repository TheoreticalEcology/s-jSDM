library(sjSDM)
library(DHARMa)
library(mltools)
library(data.table)


## simulate community:
com = simulate_SDM(env = 2L, species = 7L, sites = 100L)

## fit model:
model = sjSDM(Y = com$response,
              env = linear(com$env_weights, ~.), 
              iter = 50L,
              family = binomial("probit"))
# increase iter for your own data 
sims = simulate(model, nsim = 200L)
dim(sims)

sp = 1
dim(sims)
sim = 
  createDHARMa(t(array(sims, c(200, 100*7))), 
              c(com$response), 
              c(predict(model)),
              integerResponse = TRUE, method = 'traditional')
plot(sim)



vs = 
sapply(1:100, function(i) {
  empirical_cdf(as.data.table(sims[,i,]) + matrix(runif(200*7, -0.5, 0.5), 200, 7),
              ubounds = as.data.table(com$response[i,]+runif(7,-0.5,0.5)))$CDF
})


sim$scaledResiduals = c(vs)
sim$observedResponse = c(com$response)
sim$fittedPredictedResponse = c(predict(model))
sim$nObs = 700
plot(sim)


# for (i in 1:n) {
#   minSim <- mean(simulations[i, ] < observed[i])
#   maxSim <- mean(simulations[i, ] <= observed[i])
#   if (minSim == maxSim) 
#     scaledResiduals[i] = minSim
#   else scaledResiduals[i] = runif(1, minSim, maxSim)
# }