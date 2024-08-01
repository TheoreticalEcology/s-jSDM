\dontrun{
library(sjSDM)
# simulate community:
community = simulate_SDM(env = 3L, species = 10L, sites = 100L)

Occ <- community$response
Env <- community$env_weights
SP <- data.frame(matrix(rnorm(200, 0, 0.3), 100, 2)) # spatial coordinates


# fit model:
model <- sjSDM(Y = Occ, 
               env = linear(data = Env, formula = ~X1+X2+X3), 
               spatial = linear(data = SP, formula = ~0+X1*X2), 
               family=binomial("probit"),
               verbose = FALSE,
               iter = 20) # increase iter for real analysis

# Calculate ANOVA for env, space, associations, for details see ?anova.sjSDM
an = anova(model, samples = 10, verbose = FALSE) # increase iter for real analysis

# Show anova fractions
plot(an)

# ANOVA tables with different way to handle fractions
summary(an)
summary(an, fractions = "discard")
summary(an, fractions = "proportional")
summary(an, fractions = "equal")

# Internal structure
int = internalStructure(an, fractions = "proportional")

print(int)

plot(int) # default is negative values will be set to 0
plot(int, negatives = "scale") # global rescaling of all values to range 0-1
plot(int, negatives = "raw") # negative values will be discarded

plotAssemblyEffects(int)
plotAssemblyEffects(int, negatives = "floor")
plotAssemblyEffects(int, response = "sites", pred = as.factor(c(rep(1, 50), rep(2, 50))))
plotAssemblyEffects(int, response = "species", pred = runif(10))
plotAssemblyEffects(int, response = "species", pred = as.factor(c(rep(1, 5), rep(2, 5))))
}
