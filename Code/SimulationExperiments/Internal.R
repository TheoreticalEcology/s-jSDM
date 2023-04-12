library(sjSDM)
torch = sjSDM:::pkg.env$torch
SP = 30

Sigma = diag(1.0, SP)
Sigma = cov2cor(rWishart(1, 30, diag(1.0, SP))[,,1])
Sigma[,15:SP] = Sigma[15:SP,] = 0
diag(Sigma) = 1.0
beta = rep(0.0, 30)
Env = rnorm(1000)
Y = 1*((Env %*% t(beta) + mvtnorm::rmvnorm(1000, sigma = Sigma))>0)

XY = matrix(rnorm(2000), 1000, 2)
model = sjSDM(Y = Y, 
              env = linear(matrix(Env, ncol = 1L)), 
              spatial = linear(XY, ~0+X1:X2), biotic = bioticStruct(diag = FALSE),
              iter = 100L) # increase iter for your own data 
sjSDM:::pkg.env$fa$MVP_logLik(Y, predict(model, type = "raw"), model$model$sigma, device = torch$device("cpu"))
sum(logLik(model, individual = TRUE)[[1]])

model = sjSDM(Y = Y, 
              env = linear(matrix(Env, ncol = 1L)), 
              spatial = linear(XY, ~0+X1:X2), biotic = bioticStruct(diag = TRUE),
              iter = 100L) # increase iter for your own data 
sjSDM:::pkg.env$fa$MVP_logLik(Y, predict(model, type = "raw"), diag(1.0, 30), device = torch$device("cpu"))
sum(logLik(model, individual = TRUE)[[1]])

model1 = sjSDM(Y = Y, 
              env = linear(matrix(Env, ncol = 1L), ~0), 
              spatial = linear(XY, ~0+X1:X2), biotic = bioticStruct(diag = FALSE),
              iter = 100L) # increase iter for your own data 
sjSDM:::pkg.env$fa$MVP_logLik(Y, predict(model1, type = "raw"), model1$model$sigma, device = torch$device("cpu"))
tmp = sapply(1:1000, function(i) colSums(logLik(model1, individual = TRUE)[[1]]))


model = sjSDM(Y = Y, 
              env = linear(matrix(Env, ncol = 1L), ~.), 
              spatial = linear(XY, ~0+X1:X2), biotic = bioticStruct(diag = FALSE),
              iter = 100L) # increase iter for your own data 
sjSDM:::pkg.env$fa$MVP_logLik(Y, predict(model2, type = "raw"), model2$model$sigma, device = torch$device("cpu"))
sum(logLik(model2, individual = TRUE)[[1]])

sjSDM:::pkg.env$fa$MVP_logLik(Y, matrix(0, 1000, 30), model$model$sigma, device = torch$device("cpu"), internal = TRUE)$sum()
sjSDM:::pkg.env$fa$MVP_logLik(Y, matrix(0, 1000, 30), diag(1.0, 30), device = torch$device("cpu"), internal = TRUE)$sum()

sjSDM:::pkg.env$fa$MVP_logLik(Y[,1:15], matrix(0, 1000, 15), chol(Sigma[1:15,1:15]), device = torch$device("cpu"), internal = TRUE)$sum(0L)
sjSDM:::pkg.env$fa$MVP_logLik(Y[,1:15], matrix(0, 1000, 15), diag(1.0, 15), device = torch$device("cpu"), internal = TRUE)$sum(0L)

sjSDM:::pkg.env$fa$MVP_logLik(Y[,16:30], matrix(0, 1000, 15), chol(Sigma[16:30,16:30]), device = torch$device("cpu"), internal = TRUE)$sum(0L)
sjSDM:::pkg.env$fa$MVP_logLik(Y[,16:30], matrix(0, 1000, 15), diag(1.0, 15), device = torch$device("cpu"), internal = TRUE)$sum(0L)


SP = 10
Sigma = diag(1.0, SP)
Sigma = cov2cor(rWishart(1, SP, diag(1.0, SP))[,,1])
Sigma[,1:5] = Sigma[1:5,] = 0.0
diag(Sigma) = 1.0
beta = rep(0.3, SP)
Env = rnorm(1000)
Y = 1*((Env %*% t(beta) + mvtnorm::rmvnorm(1000, sigma = Sigma))>0)

torch = sjSDM:::pkg.env$torch

colSums(
(py_to_r(sjSDM:::pkg.env$fa$MVP_logLik(Y, matrix(0, 1000, SP), chol(Sigma), device = torch$device("cpu"), internal = TRUE)) -
  py_to_r(sjSDM:::pkg.env$fa$MVP_logLik(Y, matrix(0, 1000, SP), diag(1.0, SP), device = torch$device("cpu"), internal = TRUE)))
)

sum(py_to_r(sjSDM:::pkg.env$fa$MVP_logLik(Y, matrix(0, 1000, SP), chol(Sigma), device = torch$device("cpu"), internal = TRUE)))

sp = 15L
ind = 16:30
torch = sjSDM:::pkg.env$torch
DT = torch$float32
Ys = torch$tensor(Y[,ind], dtype=DT)
sigma = torch$tensor(chol(Sigma[ind,ind]), dtype = DT)
pred = torch$zeros(list(1000L, sp))
noise = torch$randn(list(100L, 1000L, sp))
E = torch$distributions$Normal(0.0, 1.0)$cdf(torch$tensordot(noise, sigma$t(), dims = 1L))$mul(torch$tensor(0.999999))$add(torch$tensor(0.0000005))
logprob = E$log()$mul(Ys)$add(torch$ones(1L)$sub(E)$log()$mul(torch$ones(1L)$sub(Ys)) )$neg()$sum(2L)$neg()
logprob$sum()
logprob2 = E$log()$mul(Ys)$add(torch$ones(1L)$sub(E)$log()$mul(torch$ones(1L)$sub(Ys)) )
Prop = logprob2$exp()$mean(0L)$log()$abs()
Prop$sum()
Prop = Prop$multiply( (torch$ones(1L)$div(Prop$sum(dim=1L)) )$reshape(list(-1L, 1L))$repeat_interleave(Ys$shape[1L],1L) )
Prop$sum()
maxlogprob = logprob$max(dim = 0L)$values
Eprob = logprob$sub(maxlogprob)$exp()$mean(dim = 0L)
Eprob = Eprob$log()$neg()$sub(maxlogprob)$reshape(list(-1L,1L))$repeat_interleave(Ys$shape[1L],1L)$mul(Prop)
Eprob$sum(0L)
#   logprob = E.log().mul(y).add((1.0 - E).log().mul(1.0 - y)).neg().sum(dim = 2).neg()

XX = mvtnorm::rmvnorm(1000, sigma = Sigma)
log(apply(dbinom(Y[3,,drop=FALSE],1, pnorm(XX)), 2, mean))


YY = mvtnorm::rmvnorm(1000, sigma = Sigma)
sum(mvtnorm::dmvnorm(YY, log = TRUE))
sum(mvtnorm::dmvnorm(YY, log = TRUE, sigma = Sigma))

hmsc = list()
studyDesign = data.frame(sample = as.factor(1:nrow(Y)))
rL = HmscRandomLevel(units = studyDesign$sample)
model = Hmsc(Y = Y, XData = data.frame(E = Env), XFormula = ~1+E,
             studyDesign = studyDesign, ranLevels = list(sample = rL), distr = "probit")
time =
  system.time({
    model = sampleMcmc(model, thin = 50, samples = 100, transient = 100, verbose = 50,
                       nChains = 1L)
  })
rr = Hmsc::computeVariancePartitioning(model)

E = logLik( sjSDM(Y = Y, 
                   env = linear(matrix(Env, ncol = 1L), ~.), 
                   spatial = linear(XY, ~0), biotic = bioticStruct(diag = TRUE),
                   iter = 100L,family = binomial(link = "probit")), individual = TRUE) # increase iter for your own data )

EB = logLik( sjSDM(Y = Y, 
                   env = linear(matrix(Env, ncol = 1L), ~.), 
                   spatial = linear(XY, ~0), biotic = bioticStruct(diag = FALSE),
                   iter = 100L,family = binomial(link = "probit")), individual = TRUE)
B = logLik( sjSDM(Y = Y, 
                   env = linear(matrix(Env, ncol = 1L), ~0), 
                   spatial = linear(XY, ~0), biotic = bioticStruct(diag = FALSE),
                   iter = 100L,family = binomial(link = "probit")), individual = TRUE)
Null = logLik( sjSDM(Y = Y, 
                     env = linear(matrix(Env, ncol = 1L), ~0), 
                     spatial = linear(XY, ~0), biotic = bioticStruct(diag = TRUE),family = binomial(link = "probit"),
                     iter = 1L), individual = TRUE)
ff =function(a, b) 1 - (b/a)

NN = colSums(Null[[1]])
E = colSums(E[[1]])
EB = colSums(EB[[1]])
B = colSums(B[[1]])

 (ff(-NN, -EB) - ff(-NN, -E))

(ff(-NN, -EB) - ff(-NN, -B))/2


B + (EB-E-B)

colSums(R1[[1]]) - colSums(R2[[1]])

torch = sjSDM:::pkg.env$torch
sjSDM:::pkg.env$fa$MVP_logLik(Y, matrix(0.0, 1000L, 30L), sigma = diag(1.0, 30), device = torch$device("cpu"))
sjSDM:::pkg.env$fa$MVP_logLik(Y, matrix(0.0, 1000L, 30L), sigma = Sigma, device = torch$device("cpu"))

KK = M$model$logLik(X = matrix(0, nrow = 1000L, 0L), Y = Y, SP = matrix(0, nrow = 1000L, 0L),individual = TRUE)

summary(model)
an = anova(model)
RR = plot(an, internal = TRUE,type = "Nagelkerke")
an$sites$R2_McFadden$A

predict(model, newdata = com$env_weights, SP = XY)
R2 = Rsquared(model)
print(R2)

## Using spatial eigenvectors as predictors to account 
## for spatial autocorrelation is a common approach:
SPV = generateSpatialEV(XY)
model = sjSDM(Y = com$response, env = linear(com$env_weights), 
              spatial = linear(SPV, ~0+., lambda = 0.1),
              iter = 50L) # increase iter for your own data 
summary(model)




###### negative binomial
  (A1 = mean(rnbinom(10000, size = 20, prob = 0.1)))
A2 = rnbinom(10000, mu = 20, size = 2)
# prob = mu/(size+mu)
torch = sjSDM:::pkg.env$torch
sjSDM:::pkg.env$torch$distributions$NegativeBinomial(torch$tensor(0.5), 
                                                     probs = torch$tensor(.9))$log_pro

prob = 20/(20+20)
mu/(size+mu)

1/2.2


SP = 10
Sigma = diag(1.0, SP)
Sigma[] = 0.9
Sigma[1:5,] = Sigma[,1:5] = 0.0
diag(Sigma) = 1.0
fields::image.plot(Sigma)
beta = c(rep(1.5, 5), rep(0.3, 5))
Env = rnorm(500)
XY = matrix(rnorm(1000), 500, 2)
betaSP = rep(0.0, SP)
Y = round(exp((Env %*% t(beta) + (XY[,1,drop=FALSE]*XY[,2,drop=FALSE]) %*% t(betaSP) +  mvtnorm::rmvnorm(500, sigma = Sigma))))

m = sjSDM(Y, matrix(Env, ncol = 1), spatial = linear(XY, ~0+.), family = "nbinom", 
          learning_rate = 0.01, iter = 100L,
          control = sjSDMControl(RMSprop(weight_decay = 0.00001)))
an = anova(m)

plot(predict(m, type = "link")[,1], Y[,1])

plot(predict(glmmTMB(Y[,8]~Env, family = nbinom1()), type = "response"), Y[,8])
m$model$theta

Y = predict(m)[,10]
theta = reticulate::py_to_r( m$model$theta$data$cpu()$numpy() )[1]
theta
obs = m$data$Y[,1]

sum(dnbinom(obs, Y, Y/(Y+theta), log = TRUE))

disps[1]

sum(dnbinom(m$data$Y[,9], disps[9], mu = Y, log = TRUE))

Y = predict(m)[,9]
eps = torch$tensor(0.00001)
one = torch$tensor(1.0)
E = torch$tensor(Y)
Ys = torch$tensor(m$data$Y[,9])

theta = one$div( torch$nn$functional$softplus(torch$tensor(m$theta[10]) )$add(eps) )
theta
probs = one$sub( theta$div(E$add(theta)) )$add(eps)

sjSDM:::pkg.env$torch$distributions$NegativeBinomial(theta, 
                                                     probs = probs)$log_prob(torch$tensor(YS))$sum()

theta = 1/(softplus(m$theta[9])+0.00001)
sum(dnbinom(m$data$Y[,9], size =  theta,prob =  theta / (theta+Y) +0.00001 , log = TRUE))
sum(dnbinom(m$data$Y[,9], size =  theta,prob =  theta / (theta+Y) +0.00001 , log = TRUE))

# mu
# theta = 1/(softplus(theta)+0.0001)
# probs = (1 - (theta/theta+mu))+0.0001
# total_count = theta, probs = probs
# y_hat = tf$exp(y_pred)
# theta = tf$div(k_constant(1.0, k_floatx()),(k_softplus(theta_0) + eps))
# probs = k_constant(1., k_floatx()) - tf$div(theta , (theta+y_hat))+ eps
# final = dist$NegativeBinomial(total_count = theta, probs = probs)$log_prob(y_true)
return(k_mean(-final))

torch$ones(1L)$div(torch$nn$functional$softplus(m$model$theta)$add(torch$tensor(0.00001)))

torch$ones(1L)$add(torch$nn$functional$softplus(m$model$theta)$add(torch$tensor(0.00001)))




library(sjSDM)

model = simulate_SDM(env = 1)
fitted_model = sjSDM(Y = model$response, model$env_weights, se = TRUE)

# No species names
plot(fitted_model)

# Make species names
fitted_model$species = paste("Sp", letters[1:5])
plot(fitted_model)

