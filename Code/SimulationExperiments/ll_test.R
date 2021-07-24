library(parzer)
get_coords = function(lat, lon) {
  symbs = c("°", "'", "''")
  latP = paste0(sapply(1:3, function(i) {
    paste0((as.list(parzer::parse_parts_lat(paste0(lat))))[[i]],symbs[i])
  }), collapse = "")
  
  lonP = paste0(sapply(1:3, function(i) {
    paste0((as.list(parzer::parse_parts_lat(paste0(lon))))[[i]],symbs[i])
  }), collapse = "")
  
  return(paste0(latP, "N ", lonP, "E"))
}

#### Load data, adjust path ####
data = read.csv("../../../Downloads/0280201-200613084148143/occurrence.txt", sep = "\t")

#### Available species ####
sort(unique(data$species))

#### Select one species and parse coordinates ####
Pilz = data[data$species == "Suillus reticulatus",]
k = 1
get_coords(Pilz$decimalLatitude[k], Pilz$decimalLongitude[k])



library(sjSDM)
library(gllvm)

com1 = simulate_SDM(species = 300L, sites = 500L, wishart = TRUE)
com2 = simulate_SDM(species = 300L, sites = 500L,wishart = FALSE)


m1 = sjSDM(com1$response, com1$env_weights, sampling = 100L)
m2 = sjSDM(com2$response, com2$env_weights, sampling = 100L)

g1 = gllvm(com1$response, data.frame(com1$env_weights), formula = ~., family = binomial("probit"))
g2 = gllvm(com2$response, data.frame(com2$env_weights), formula = ~., family = binomial("probit"))

com1$corr_acc(getCov(m1))
com1$corr_acc(getResidualCov(g1)$cov)

com2$corr_acc(getCov(m2))
com2$corr_acc(getResidualCov(g2)$cov)

CC = trialr::rlkjcorr(1, 10L)
cor2cov(CC)

diag(CC) * CC * diag(CC)


SS = rWishart(1, 5, diag(1., 5))[,,1]
D = sqrt(diag(diag(SS)))
SS2 = D %*% cov2cor(SS) %*% D


library(sjSDM)
com = simulate_SDM(env = 3L, species = 20L, sites = 100L)

model = sjSDM(Y = com$response,env = com$env_weights, iter = 70L) 
true = rbind(0, com$species_weights)
V = sqrt(diag(getCov(model)) - 1)
coef = t(coef(model)[[1]])
coef

r2 = function(a, b) mean(sqrt( (a-b)^2 ))
V = sqrt(diag(getCov(model)) - 1)
r2(coef / matrix(V, ncol = ncol(com$response), nrow = nrow(coef), byrow = TRUE), true)
r2(coef , true)



U = diag(2*com$response[1,]-1)
U %*% (com$correlation - diag(1., 20L)) %*% U

com = simulate_SDM(env = 3L, species = 7L, sites = 100L)

## fit model:
model = sjSDM(Y = com$response[1:2,],env = DNN(com$env_weights[1:2,,drop=FALSE], hidden = NULL), iter = 2L) 

success = 
tryCatch({
  cat(error) 
  # your nested loop
  
  
}, error = function(e) e)
class(success)
if(inherits(success, "error")) { 
  sink(file = paste0(runif(1, 1e5, 1e6), ".txt")) 
  print("Hello")
  sink()
  }
  



com = simulate_SDM(env = 3L, species = 7L, sites = 500L)

## fit model:
model = sjSDM(Y = com$response,env = com$env_weights, iter = 100L) 
model2 = sjSDM(Y = com$response,env = com$env_weights,biotic = bioticStruct(lambda = 5, alpha = 0), iter = 100L) 


remotes::install_github("kcf-jackson/typeChecker")
library(typeChecker)

f = function(a = ? integer){print(a)}

f(a="Edwad")
f(5)

dog <- function(name = ? character) {
  list(name = name)
}

dog("Archie")   # correct usage
dog(123) 


library(checkmate)
assertCharacter(2)
asser



library(sjSDM)
?sjSDM
com = simulate_SDM()
XY = matrix(rnorm(200), 100, 2)
SPV = generateSpatialEV(XY)
com = simulate_SDM(env = 3L, species = 7L, sites = 100L)
model = sjSDM(Y = com$response, env = DNN(com$env_weights, dropout = 0.5, hidden = as.integer(rep(30L, 5))), 
              spatial = DNN(SPV,hidden = c(15L, 15L), dropout = 0.5, ~0+.),
              iter = 1L, step_size = 100L) # increase iter for your own data 



preds1 = abind::abind(lapply(1:500, function(i) predict(model, dropout = FALSE)), along = 0L)
preds2 = abind::abind(lapply(1:500, function(i) predict(model, dropout = TRUE)), along = 0L)

par(mfrow = c(1, 2))
plot(density(preds1[,1,1]), xlim = c(0, 1))
plot(density(preds2[,1,1]), xlim = c(0, 1))

abline(v = mean(preds2[,1,1]) + 2*sd(preds2[,1,1]))
abline(v = mean(preds2[,1,1]) - 2*sd(preds2[,1,1]))
abline(v = quantile(preds2[,1,1], c(0.1, 0.9)), col = "red")


m = lme4::lmer(Y~X+(1|as.factor(G)))
summary(m)
opt2 = function(par, RE2) {
pred = par[1]*X + par[2] +RE2[G]
-sum(dnorm(Y, pred, sd = exp(par[3]), log = TRUE)) - sum(dnorm(RE2,0,sd = exp(par[4]), log = TRUE))
}
opt1 = function(par, par2) {
pred = par2[1]*X + par2[2] + par[G]
-sum(dnorm(Y, pred, sd = exp(par2[3]), log = TRUE)) - sum(dnorm(par,0,sd = exp(par2[4]), log = TRUE))
}
RE2 = rnorm(g, 0, 0.1)
par = rnorm(4, 0, 0.1)
for(i in 1:100) {
o1 = optim(par, opt2, RE2 = RE2)
par = o1$par
o2 = optim(RE2, opt1, par2 = par)
RE2 = o2$par
}
cor(RE, ranef(m)[[1]][,1])
cor(RE, RE2)
length85:15
length(5:15)
length(5:14)
opt3 = function(par) {
pred = par[1]*X + par[2] +par[5:14][G]
-sum(dnorm(Y, pred, sd = exp(par[3]), log = TRUE)) - sum(dnorm(par[5:14],0,sd = exp(par[4]), log = TRUE))
}
o3 = optim(rnorm(15, 0, 0.1), opt3)
cor(RE, o3$par[5:14])
cor(RE, ranef(m)[[1]][,1])
cor(RE, RE2)
indices = sample.int(100, 10L)
G
indices = sample.int(100, 10L)
pred = par[1]*X[indices] + par[2][indices] +RE2[G[indices]]
-sum(dnorm(Y, pred, sd = exp(par[3]), log = TRUE)) - sum(dnorm(RE2,0,sd = exp(par[4]), log = TRUE))
-sum(dnorm(Y[indices], pred, sd = exp(par[3]), log = TRUE)) - sum(dnorm(RE2,0,sd = exp(par[4]), log = TRUE))
Y[indices]
pred
par[1]*X[indices]
G[indices]
RE2
RE2[G[indices]]
opt2 = function(par, RE2) {
pred = par[1]*X[indices] + par[2] +RE2[G[indices]]
-sum(dnorm(Y[indices], pred, sd = exp(par[3]), log = TRUE)) - sum(dnorm(RE2,0,sd = exp(par[4]), log = TRUE))
}
opt1 = function(par, par2) {
pred = par2[1]*X[indices] + par2[2] + par[G[indices]]
-sum(dnorm(Y[indices], pred, sd = exp(par2[3]), log = TRUE)) - sum(dnorm(par,0,sd = exp(par2[4]), log = TRUE))
}
RE2 = rnorm(g, 0, 0.1)
par = rnorm(4, 0, 0.1)
for(i in 1:100) {
indices = sample.int(100, 10L)
o1 = optim(par, opt2, RE2 = RE2)
par = o1$par
o2 = optim(RE2, opt1, par2 = par)
RE2 = o2$par
}
cor(RE, ranef(m)[[1]][,1])
cor(RE, RE2)
for(i in 1:100) {
indices = sample.int(100, 10L)
o1 = optim(par, opt2, RE2 = RE2)
par = o1$par
o2 = optim(RE2, opt1, par2 = par)
RE2 = o2$par
}
cor(RE, RE2)
RE2 = rnorm(g, 0, 0.1)
par = rnorm(4, 0, 0.1)
for(i in 1:100) {
indices = sample.int(100, 50L)
o1 = optim(par, opt2, RE2 = RE2)
par = o1$par
o2 = optim(RE2, opt1, par2 = par)
RE2 = o2$par
}
cor(RE, ranef(m)[[1]][,1])
cor(RE, RE2)
RE2 = rnorm(g, 0, 0.1)
par = rnorm(4, 0, 0.1)
for(i in 1:100) {
indices = sample.int(100, 100L)
o1 = optim(par, opt2, RE2 = RE2)
par = o1$par
o2 = optim(RE2, opt1, par2 = par)
RE2 = o2$par
}
cor(RE, ranef(m)[[1]][,1])
cor(RE, RE2)
RE2 = rnorm(g, 0, 0.1)
par = rnorm(4, 0, 0.1)
for(i in 1:100) {
indices = sample.int(100, 100L)
o1 = optim(par, opt2, RE2 = RE2)
par = o1$par
o2 = optim(RE2, opt1, par2 = par)
RE2 = o2$par
}
cor(RE, ranef(m)[[1]][,1])
cor(RE, RE2)
G
indices
RE2 = rnorm(g, 0, 0.1)
par = rnorm(4, 0, 0.1)
for(i in 1:100) {
indices = sample.int(100, 100L)
o1 = optim(par, opt2, RE2 = RE2)
par = o1$par
o2 = optim(RE2, opt1, par2 = par)
RE2 = o2$par
}
cor(RE, ranef(m)[[1]][,1])
cor(RE, RE2)
RE2 = rnorm(g, 0, 0.1)
par = rnorm(4, 0, 0.1)
for(i in 1:100) {
indices = 1:100
o1 = optim(par, opt2, RE2 = RE2)
par = o1$par
o2 = optim(RE2, opt1, par2 = par)
RE2 = o2$par
}
cor(RE, ranef(m)[[1]][,1])
cor(RE, RE2)
RE
RE2
opt2 = function(par, RE2) {
pred = par[1]*X + par[2] +RE2[G]
-sum(dnorm(Y, pred, sd = exp(par[3]), log = TRUE)) - sum(dnorm(RE2,0,sd = exp(par[4]), log = TRUE))
}
opt1 = function(par, par2) {
pred = par2[1]*X + par2[2] + par[G]
-sum(dnorm(Y, pred, sd = exp(par2[3]), log = TRUE)) - sum(dnorm(par,0,sd = exp(par2[4]), log = TRUE))
}
RE2 = rnorm(g, 0, 0.1)
par = rnorm(4, 0, 0.1)
for(i in 1:100) {
indices = sample.int(100, 10L)
o1 = optim(par, opt2, RE2 = RE2)
par = o1$par
o2 = optim(RE2, opt1, par2 = par)
RE2 = o2$par
}
cor(RE, ranef(m)[[1]][,1])
cor(RE, RE2)
RE2
RE2[G]
opt2 = function(par, RE2) {
pred = par[1]*X[indices] + par[2] +RE2[G][indices]
-sum(dnorm(Y[indices], pred, sd = exp(par[3]), log = TRUE)) - sum(dnorm(RE2,0,sd = exp(par[4]), log = TRUE))
}
opt1 = function(par, par2) {
pred = par2[1]*X[indices] + par2[2] + par[G][indices]
-sum(dnorm(Y[indices], pred, sd = exp(par2[3]), log = TRUE)) - sum(dnorm(par,0,sd = exp(par2[4]), log = TRUE))
}
RE2 = rnorm(g, 0, 0.1)
par = rnorm(4, 0, 0.1)
for(i in 1:100) {
indices = sample.int(100, 10L)
o1 = optim(par, opt2, RE2 = RE2)
par = o1$par
o2 = optim(RE2, opt1, par2 = par)
RE2 = o2$par
}
cor(RE, ranef(m)[[1]][,1])
cor(RE, RE2)
RE2 = rnorm(g, 0, 0.1)
par = rnorm(4, 0, 0.1)
for(i in 1:100) {
indices = sample.int(100, 100L)
o1 = optim(par, opt2, RE2 = RE2)
par = o1$par
o2 = optim(RE2, opt1, par2 = par)
RE2 = o2$par
}
cor(RE, ranef(m)[[1]][,1])
cor(RE, RE2)
opt2 = function(par, RE2, indices) {
pred = par[1]*X[indices] + par[2] +RE2[G][indices]
-sum(dnorm(Y[indices], pred, sd = exp(par[3]), log = TRUE)) - sum(dnorm(RE2,0,sd = exp(par[4]), log = TRUE))
}
opt1 = function(par, par2, indices) {
pred = par2[1]*X[indices] + par2[2] + par[G][indices]
-sum(dnorm(Y[indices], pred, sd = exp(par2[3]), log = TRUE)) - sum(dnorm(par,0,sd = exp(par2[4]), log = TRUE))
}
RE2 = rnorm(g, 0, 0.1)
par = rnorm(4, 0, 0.1)
for(i in 1:100) {
indices = sample.int(100, 100L)
o1 = optim(par, opt2, RE2 = RE2, indices = indices)
par = o1$par
o2 = optim(RE2, opt1, par2 = par, indices = indices)
RE2 = o2$par
}
cor(RE, ranef(m)[[1]][,1])
cor(RE, RE2)



library(sjSDM)
com = simulate_SDM(sparse = 0.95, species = 20L)
m1 = sjSDM(com$response[1:50,], linear(com$env_weights[1:50,], ~0+.), biotic = bioticStruct(lambda = 0.00, alpha = 0.1))
m2 = sjSDM(com$response[1:50,], linear(com$env_weights[1:50,], ~0+.), biotic = bioticStruct(lambda = 0.01, alpha = 0.1))
mean(sapply(1:20, function(i) m1$model$logLik(com$env_weights[51:100,], com$response[51:100,], sampling = 1000L)[[1]]))
mean(sapply(1:20, function(i) m2$model$logLik(com$env_weights[51:100,], com$response[51:100,], sampling = 1000L)[[1]]))


sqrt(mean((t(m$model$env_weights[[1]]) - com$species_weights)**2))
fa$MVP_logLik(com$response[51:100,], com$env_weights[51:100,] %*% t(m1$model$env_weights[[1]]), sigma = m$sigma, device = torch$device("cpu"), sampling = 50000L)
fa$MVP_logLik(com$response[51:100,], com$env_weights[51:100,] %*% t(m2$model$env_weights[[1]]), sigma = m$sigma, device = torch$device("cpu"), sampling = 50000L)

