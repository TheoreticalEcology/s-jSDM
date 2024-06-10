\dontrun{

library(sjSDM)
# simulate community:
com = simulate_SDM(env = 3L, species = 10L, sites = 100L)
XY = matrix(runif(100, -1, 1), 100, 2)

# fit model:
model = sjSDM(Y = com$response,env = com$env_weights,spatial = linear(XY, ~0+.), iter = 50L) 
# increase iter for your own data 

# Anova 
an = anova(model)
plot(an)
an # note this table is sequential (type I) ... other shares have to be 
# calculated by hand

# Internal structure
plotInternalStructure(an)

plotAssemblyEffects(an)


an1 = anova(model, fractions = "equal")
an2 = anova(model, fractions = "proportional")





plotInternalStructure(an2, add_shared = FALSE)
plotAssemblyEffects(an2)
plotAssemblyEffects(an2, env = runif(100))
plotAssemblyEffects(an2, env = as.factor(c(rep(1, 50), rep(2, 50))))
plotAssemblyEffects(an2, trait = runif(10))
plotAssemblyEffects(an2, trait = as.factor(c(rep(1, 5), rep(2, 5))))


}