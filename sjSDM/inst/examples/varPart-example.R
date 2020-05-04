library(sjSDM)


com = simulate_SDM(sites = 300L, species = 12L)
SP = matrix(rnorm(300*2), 300, 2)
model = sjSDM(Y = com$response,env = linear(com$env_weights, lambda = 0.001), spatial = linear(SP, lambda = 0.001), biotic = bioticStruct(lambda = 0.001),iter = 40L)
varPart(model)
# or another type based on R^2
varPart(model, method = "III", order = c("ESB", "ES", "E"))

if(require(gllvm)){
# varPart with method="coef" can be also used for other models: 
## we can provide the spatial terms as normal predictors (note: you should use SP1:SP2 )
lvm = gllvm(com$response, data.frame(com$env_weights, SP), family = binomial("probit"))
model2 = list(t(cbind(coef(lvm)$Xcoef, coef(lvm)$Intercept)), getResidualCov(lvm, FALSE)$cov, cov(lvm$TMBfn$env$data$x))
varPart(model2)
}