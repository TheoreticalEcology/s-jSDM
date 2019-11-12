if(version$minor > 5) RNGkind(sample.kind="Rounding")
library(deepJSDM)
sites = c(50, 70, 100, 140, 180, 260, 320, 400, 500)
species = c(0.1, 0.2, 0.3, 0.4,0.5)
env = c(3,5,7)

setup = expand.grid(sites, species, env)
colnames(setup) = c("sites", "species", "env")
setup = setup[order(setup$sites,decreasing = FALSE),]

set.seed(42)
data_sets = vector("list", nrow(setup))
counter = 1L
for(i in 1:nrow(setup)) {
  for(j in 1:10){

    tmp = setup[i,]
    sim = simulate_SDM(env = tmp$env,sites = 2*tmp$sites,species = as.integer(tmp$species*tmp$sites))
    X = sim$env_weights
    Y = sim$response
    
    ### split into train and test ###
    indices = sample.int(nrow(X), 0.5*nrow(X))
    train_X = X[indices, ]
    train_Y = Y[indices, ]
    test_X = X[-indices, ]
    test_Y = Y[-indices, ]
    
    data_sets[[counter]] = list(setup = tmp, X = X, Y = Y, train_X = train_X, train_Y = train_Y, test_X = test_X, test_Y, test_Y, sim = sim)
    counter = counter + 1L
  }
  
}
save(data_sets, setup, file = "data_sets.RData")