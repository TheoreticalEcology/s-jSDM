\dontrun{
library(sjSDM)
com = simulate_SDM(sites = 300L, species = 12L, 
                   link = "identical", response = "identical")
Raw = com$response
SP = matrix(rnorm(300*2), 300, 2)
SPweights = matrix(rnorm(12L), 1L)
SPweights[1,1:6] = 0
Y = Raw + (SP[,1,drop=FALSE]*SP[,2,drop=FALSE]) %*% SPweights
Y = ifelse(Y > 0, 1, 0)

model = sjSDM(Y = Y,env = linear(com$env_weights, lambda = 0.001), 
              spatial = linear(SP,formula = ~0+X1:X2, lambda = 0.001), 
              biotic = bioticStruct(lambda = 0.001),iter = 40L)
imp = importance(model)
plot(imp)
}