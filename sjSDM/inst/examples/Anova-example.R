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

print(an, fractions = "equal")
print(an, fractions = "proportional")
print(an, fractions = "discard")




# Internal structure
plotInternalStructure(an, fractions = "discard")

plotAssemblyEffects(an)
plotAssemblyEffects(an, env = runif(100))
plotAssemblyEffects(an, env = as.factor(c(rep(1, 50), rep(2, 50))))
plotAssemblyEffects(an, trait = runif(10))
plotAssemblyEffects(an, trait = as.factor(c(rep(1, 5), rep(2, 5))))


}