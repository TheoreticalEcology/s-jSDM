library(sjSDM)
library(sjSDM)
SP = 10

set.seed(42)
Sigma = diag(1.0, SP)
Sigma = cov2cor(rWishart(1, SP, diag(1.0, SP))[,,1])
Sigma[,1:5] = Sigma[5:1,] = 0
diag(Sigma) = 1.0
beta = c(rep(1.5, 5), rep(0.5, 5))
Env = rnorm(1000)
Y = 1*((Env %*% t(beta) + mvtnorm::rmvnorm(1000, sigma = Sigma))>0)
XY = matrix(rnorm(2000), 1000, 2)

model = sjSDM(Y = Y, 
              env = linear(matrix(Env, ncol = 1L)), 
              spatial = linear(XY, ~0+X1:X2),
              iter = 100L) # increase iter for your own data 
an = anova(model)
RR = plot(an, internal = TRUE)
an$TypeIII


set.seed(1)
XY = matrix(rnorm(2000), 1000, 2)
SP = 10
Sigma = diag(1.0, SP)
Sigma = cov2cor(rWishart(1, SP, diag(1.0, SP))[,,1])
Sigma[,1:5] = Sigma[1:5,] = 0.0
diag(Sigma) = 1.0
beta = rep(1.0, SP)
Env = rnorm(1000)
Y = 1*((Env %*% t(beta) + mvtnorm::rmvnorm(1000, sigma = Sigma))>0)

model = sjSDM(Y = Y, 
              env = linear(matrix(Env, ncol = 1L)), 
              spatial = linear(XY, ~0+X1:X2), biotic = bioticStruct(diag = FALSE),
              iter = 100L)   
an = anova(model)
R = plot(an, internal = TRUE)
R




object = model
B_m = reticulate::py_to_r( sjSDM:::pkg.env$fa$MVP_logLik(object$data$Y,Env %*% t(beta),
                                                         (chol(cov2cor(getCov(object)))), device = sjSDM:::pkg.env$torch$device("cpu"), internal = TRUE) )
colSums(B_m)
null_m = reticulate::py_to_r( sjSDM:::pkg.env$fa$MVP_logLik(object$data$Y, matrix(0, nrow(object$data$Y), ncol(object$data$Y)),
                                                            diag(1, ncol(object$data$Y)), device = sjSDM:::pkg.env$torch$device("cpu"), internal = TRUE) )

colSums(null_m - B_m)

ff = function(Sigma, Y, PP = NULL) {
  if(is.null(PP)) PP = matrix(0, nrow(Y), ncol(Y))
  lls = 
  reticulate::py_to_r( sjSDM:::pkg.env$fa$MVP_logLik(Y, PP,
                                                     Sigma, device = sjSDM:::pkg.env$torch$device("cpu"), batch_size = 50,internal = TRUE, batch_size = 1000L, sampling = 100L) )
  
  return(colSums(lls))
}

set.seed(42)
SP = 10
Sigma = diag(1.0, SP)
#Sigma = cov2cor(rWishart(1, SP, diag(1.0, SP))[,,1])
Sigma[1,2] = Sigma[2,1] = 0.8
diag(Sigma) = 1.0
beta = rep(1.0, SP)
Env = rnorm(1000)
Y = 1*((Env %*% t(beta) + mvtnorm::rmvnorm(1000, sigma = Sigma))>0)
m = sjSDM(Y, linear(matrix(Env, ncol = 1), ~.), spatial = linear(XY, ~0+.))
colSums(logLik(m, individual = TRUE)[[1]])

an = anova(m)

barplot((ff(diag(1.0, SP), Y, predict(m, type = "raw"))) - (ff( t(chol(cov2cor(getCov(m)))) , Y,  predict(m, type = "raw"))) )
ff(diag(1.0, 10), Y)


set.seed(42)
Sigma = diag(1.0, SP)
Sigma = cov2cor(rWishart(1, 30, diag(1.0, SP))[,,1])
Sigma[,15:SP] = Sigma[15:SP,] = 0
diag(Sigma) = 1.0
beta = c(rep(0.01, 15), rep(1.0, 7), rep(0.0, 8))
Env = rnorm(1000)
Y = 1*((Env %*% t(beta) + mvtnorm::rmvnorm(1000, sigma = Sigma))>0)
XY = matrix(rnorm(2000), 1000, 2)
P = Env %*% t(beta)
barplot((ff(t(chol(Sigma)), Y, P)) - (ff(diag(1.0, SP), Y, P)))

reticulate::py_to_r( sjSDM:::pkg.env$fa$MVP_logLik(Y, P,
                                                   t(chol(Sigma)), device = sjSDM:::pkg.env$torch$device("cpu"), internal = FALSE, batch_size = 1000L, sampling = 100L) )


colSums(ff(as.matrix(Matrix::nearPD(cov2cor(getCov(model)))$mat),Y, predict(model, type = "raw")))
colSums(ff(diag(1.0, 10),Y, predict(model, type = "raw")))

set.seed(42)
XY = matrix(rnorm(2000), 1000, 2)
SP = 5
Sigma = diag(1.0, SP)
#Sigma = cov2cor(rWishart(1, SP, diag(1.0, SP))[,,1])
Sigma[1,2] = Sigma[2,1] = 0.8
Sigma[1,3] = Sigma[3,1] = 0.6
#Sigma[3:10, 3:10] = 0.0
diag(Sigma) = 1.0
beta = rep(c(0.3), SP)
Env = rnorm(1000)
Y = 1*((Env %*% t(beta) + mvtnorm::rmvnorm(1000, sigma = Sigma))>0)

model = sjSDM(Y = Y, 
              env = linear(matrix(Env, ncol = 1L)), 
              spatial = linear(XY, ~0+X1:X2), biotic = bioticStruct(diag = FALSE, df = 10L),
              iter = 100L)
an = anova(model)
R = plot(an, internal = TRUE)
R$data$Species
an$TypeIII

Sigma = getCov(model)


barplot((ff(diag(1.0, SP), Y, predict(model, type = "raw"))) - (ff( t(model$model$get_sigma) , Y,  predict(model, type = "raw"))) )


ff = function(Sigma, Y, PP = NULL, samples = 100) {
  if(is.null(PP)) PP = matrix(0, nrow(Y), ncol(Y))
  SP = ncol(Y)
  res = 
    lapply(1:1000, function(i) {
      noise = (mvtnorm::rmvnorm(samples, sigma = Sigma))
      rowSums(-sapply(1:SP, function(j) dbinom(Y[i,j], 1, pnorm(noise[,j]+PP[i,j]), log = TRUE)))  
    })
  res = abind::abind(res, along = 0L)
  
  resNN = 
    lapply(1:1000, function(i) {
      noise = (mvtnorm::rmvnorm(samples, sigma = diag(1.0, ncol(Y))))
      rowSums(-sapply(1:SP, function(j) dbinom(Y[i,j], 1, pnorm(noise[,j]+PP[i,j]), log = TRUE)))  
    })
  resNN = abind::abind(resNN, along = 0L)
  
  
  res2 = 
    lapply(1:1000, function(i) {
      noise = (mvtnorm::rmvnorm(samples, sigma = Sigma))
      -sapply(1:SP, function(j) dbinom(Y[i,j], 1, pnorm(noise[,j]+PP[i,j]), log = TRUE))  /
        rowSums(-sapply(1:SP, function(j) dbinom(Y[i,j], 1, pnorm(noise[,j]+PP[i,j]), log = TRUE)))
    })
  
  res2t = (abind::abind(res2, along = 0L))
  print(apply(res2t, c(1, 3), mean)[1,])
  KK = (1-scales::rescale(apply(res2t, c(1, 3), mean)[1,]-(1/ncol(Y)), to = c(0, 1)))
  B = (log(apply(res, 1, function(p) mean(exp(-p)))) - log(apply(resNN, 1, function(p) mean(exp(-p)))) )  * matrix(KK/sum(KK), nrow = 1000, ncol = ncol(Y), byrow = TRUE)#apply(res2t, c(1, 3), mean)
  return(sum(log(apply(resNN, 1, function(p) mean(exp(-p)))))/ncol(Y) + colSums(B))
  #return(B)
}

MVP_ll = function(Sigma, Y, PP = NULL, samples = 100) {
  if(is.null(PP)) PP = matrix(0, nrow(Y), ncol(Y)) # linear predictor
  SP = ncol(Y)
  lls = 
    lapply(1:1000, function(i) {
      noise = (mvtnorm::rmvnorm(samples, sigma = Sigma))
      # noise, species
      # sum over species -> noise 
      rowSums(-sapply(1:SP, function(j) dbinom(Y[i,j], 1, pnorm(noise[,j]+PP[i,j]), log = TRUE)))  # pnorm = probit link
    })
  lls = abind::abind(lls, along = 0L)
  
  rates = 
    lapply(1:1000, function(i) {
      noise = (mvtnorm::rmvnorm(samples, sigma = Sigma))
      -sapply(1:SP, function(j) dbinom(Y[i,j], 1, pnorm(noise[,j]+PP[i,j]), log = TRUE))  /
        rowSums(-sapply(1:SP, function(j) dbinom(Y[i,j], 1, pnorm(noise[,j]+PP[i,j]), log = TRUE)))
    })
  
  rates = (abind::abind(rates, along = 0L)) 
  return(colSums(log(apply(lls, 1, function(p) mean(exp(-p)))) * apply(rates, c(1, 3), mean))) # community ll * rates (to separate them)
}

#' MVP Approximation
#' 
#' @param Sigma Covariance matrix
#' @param Y True/observed PA
#' @param PP Linear predictor (n, sp), if null zero matrix will be used
#' @param samples Number of MC samples (per species and per site)
MVP_ll = function(Sigma, Y, PP = NULL, samples = 100) {
  if(is.null(PP)) PP = matrix(0, nrow(Y), ncol(Y)) # linear predictor
  SP = ncol(Y)
  res = 
    lapply(1:1000, function(i) {
      noise = (mvtnorm::rmvnorm(samples, sigma = Sigma))
      # noise, species
      # sum over species -> noise 
      logLiks = rowSums(-sapply(1:SP, function(j) dbinom(Y[i,j], 1, pnorm(noise[,j]+PP[i,j]), log = TRUE)))  # pnorm = probit link
      llRates=  NULL #-sapply(1:SP, function(j) dbinom(Y[i,j], 1, pnorm(noise[,j]+PP[i,j]), log = TRUE))  / logLiks
      
      return(list(logLiks, llRates))
    })
  lls = abind::abind(lapply(res, function(r) r[[1]]), along = 0L)
  #rates = (abind::abind(lapply(res, function(r) r[[2]]), along = 0L)) 
  #print(colMeans(apply(rates, c(1, 3), mean))) # print average split rates
  
  ## Average logLikelihoods over MC samples ##
  logLikelihoods = log(apply(lls, 1, function(p) mean(exp(-p)) ) )
  
  ## Average split rates over MC samples ##
  #average_rates = apply(rates, c(1, 3), mean)
  
  return((logLikelihoods)) # community ll * rates (to separate them)
}



## Simulation MVP ##
SP = 4
Sigma = diag(1.0, SP)
Sigma[1,2] = Sigma[2,1] = 0.9
diag(Sigma) = c(1.0, 1.0, 1.0, 1.0)
beta = c(0,0,0,1)
Env = rnorm(1000)
Y = 1*((Env %*% t(beta) + mvtnorm::rmvnorm(1000, sigma = Sigma))>0)
XY = matrix(rnorm(2000), 1000, 2)

## MVP loglikelihoods ##
joint_ll = (MVP_ll(Sigma, Y, (Env %*% t(beta)), samples = 500))
K = sapply(1:4, function(i) sum(joint_ll-(MVP_ll(Sigma[-i,-i], Y[,-i], (Env %*% t(beta))[,-i], samples = 500))))
K

MVP_ll(diag(1.0, 4), Y, NULL, samples = 1000)

sum(dbinom(Y, 1, 0.5, TRUE))/4


MVP_ll(Sigma, Y, NULL, samples = 1000)
MVP_ll(diag(1.0, 4), Y, NULL, samples = 1000)

sjSDM:::pkg.env$fa$MVP_logLik(Y, pred = Env%*%t(beta), sigma = t(chol(Sigma)), device = sjSDM:::pkg.env$torch$device("cpu"))


library(mvProbit)

colSums(
  log(mvProbit(cbind(Y.1,Y.2,Y.3,Y.4)~const+E, 
                  data = data.frame(Y = Y, const = rep(0, 1000), E = Env ), 
                  sigma = diag(1.0, 4), coef = c(rep(0, 12)))))

sum(mvProbitLogLik(cbind(Y.1,Y.2,Y.3,Y.4)~const+E, 
         data = data.frame(Y = Y, const = rep(0, 1000), E = Env ), 
         sigma = diag(1.0, 4), coef = c(rep(0, 12))))

R1 =
mvProbitExp(cbind(Y.1,Y.2,Y.3,Y.4)~0+E, 
               data = data.frame(Y = Y, const = rep(0, 1000), E = Env ), 
               sigma = Sigma, coef = c(0,0,0,1), cond = TRUE, algorithm = "Miwa")
RR1 = (((dbinom(Y, 1, as.matrix(R1), log = TRUE))))

R2 =
  mvProbitExp(cbind(Y.1,Y.2,Y.3,Y.4)~0+E, 
              data = data.frame(Y = Y, const = rep(0, 1000), E = Env ), 
              sigma = Sigma, coef = c(0,0,0,1), cond = TRUE, algorithm = "Miwa")
RR2 = (((dbinom(Y, 1, as.matrix(R2), log = TRUE))))

lls = (RR2-RR1)/rowSums(RR2-RR1) * (rowSums(MVP_ll(diag(1.0, 4), Y, Env %*% t(c(0,0,0,0)), samples = 1000)) - rowSums(MVP_ll(Sigma, Y, Env %*% t(beta), samples = 1000)) )

sum(colSums(lls))



sum(MVP_ll(Sigma, Y, Env %*% t(beta), samples = 1000))

(colSums(dbinom(Y, 1, as.matrix(R), log = TRUE)))/sum((colSums(dbinom(Y, 1, as.matrix(R), log = TRUE))))*sum(dbinom(Y, 1, 0.5, TRUE))


sum(dbinom(Y, 1, 0.5, TRUE))/4

R2= function(a, b) 1 - (b/a)
R2(MVP_ll(Sigma, Y, env%*%beta), MVP_ll(Sigma, Y, NULL))


x = mvtnorm::rmvnorm(1000, sigma = Sigma)
xlik = dnorm(x, log = TRUE)
mean(mvtnorm::dmvnorm(x, sigma = Sigma, log = TRUE))
colMeans(xlik)


covar = Matrix::nearPD(cov((xlik)))$mat
L = t(as.matrix(Matrix::chol(covar)))
x2 = t(apply(xlik, 1, function(a) solve(L, a)))
KK = (colMeans(x2))
KK %*% covar


library(condMVNorm)
condMVN(x, sigma, dependent.ind = 4)

(ff(Sigma,Y, predict(model, type = "raw")))
(ffold(Sigma,Y, predict(model, type = "raw")))
(ff(diag(1.0, 10),Y, predict(model, type = "raw")))


barplot(R222(ff(diag(1.0, 10),Y, Env %*% t(beta), samples = 500), ff(Sigma,Y, Env %*% t(beta), samples = 500)))
barplot(R222(ffold(diag(1.0, 10),Y, Env %*% t(beta), samples = 500), ffold(Sigma,Y, Env %*% t(beta), samples = 500)))

R222(ffold(diag(1.0, 10),Y, predict(model, type = "raw")), ffold(Sigma,Y, predict(model, type = "raw")))

R222(colSums(ff(diag(1.0, 10), Y, predict(model, type = "raw"))), colSums(ff(cov2cor(getCov(model)),Y, predict(model, type = "raw"))))
