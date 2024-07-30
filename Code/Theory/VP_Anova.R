n = 50
X1 = runif(n)
X2 = X1 + rnorm(n, sd = 0.2)
X3 = runif(n)
cor(X1, X2)

Y = X1 + X3+ rnorm(n, sd = 1.0)
df = data.frame(Y = Y, X1= X1, X2 = X2, X3 = X3)

m = lm(Y~X1+X2+X3, data = df)
summary(m)
car::Anova(m)
anova(m)
df = data.frame(Y = Y, X1= X1, X2 = X2, X3 = X3)

imp = function(df) {
  model =  lm(Y~X1+X2+X3, data= df) 
  res = rep(0, 3)
  for(i in 1:3){ 
    model_tmp = model
    model_tmp$coefficients[2:4] = model_tmp$coefficients[2:4] * diag(1.0, 3)[i,]
    pred = predict(model_tmp, newdata = df)
    res[i] = 1-sum((df$Y - pred  )**2)/sum((df$Y)**2)
  }
  return(res)
}

imp(df)


# Otso
beta = coef(m)
beta
cM = cov(cbind(0, df[,-1]))

Rs = 
sapply(2:4, function(i) {
t(beta[c(1, i)] ) %*% cM[c(1, i), c(1,i)] %*% beta[c(1, i)] 
})
Rs / sum(Rs)



Rs2 = Rs / summary(m)$coef[-1,2]
Rs2 / sum(Rs2)


relaimpo::calc.relimp(m, type = c("lmg", "last", "first", "betasq", "pratt"), rela = TRUE )

car::Anova(m)



X1 = runif(1000, -1, 1)
X2 = X1 + rnorm(1000, sd = 0.3)
X3 = runif(1000, -1 ,1 )
cor(X1, X2)
X1 = scale(X1)
X2 = scale(X2)
X3 = scale(X3)
prob = plogis(5*X1 + 5*X3)
Y = rbinom(1000, 1, prob)

m = glm(Y~X1+X2+X3, family = binomial(), data = df)
rsq::rsq.partial(m, type = "n")
car::Anova(m)
anova(m)
summary(m)
df = data.frame(Y = Y, X1= X1, X2 = X2, X3 = X3)

imp_glm = function(df) {
  model =  glm(Y~X1+X2+X3, family = binomial(), data = df)
  null = glm(Y~1, family = binomial(), data = df)
  pred_null = predict(null, newdata = df, type = "response")
  f_L = sum(dbinom(df$Y, 1, pred_null, TRUE))
  
  res = rep(0, 3)
  for(i in 1:3){ 
    model_tmp = model
    model_tmp$coefficients[2:4] = model$coefficients[2:4] * diag(1.0, 3)[i,]
    model_tmp$coefficients[1] = 0
    pred = predict(model_tmp, newdata = df, type = "response")
    f_m = sum(dbinom(df$Y, 1, pred, TRUE))
    res[i] = 1 -  f_m / f_L # pseudo Rsquared (McFadden)
  }
  res = c(res, 1- sum(dbinom(df$Y, 1, predict(model, newdata = df, type = "response"), TRUE))/f_L)
  
  return(res)
}
(imp_glm(df))


model =  glm(Y~X3, family = binomial(), data = df)
model

library(lme4)
df$site = as.factor(sample.int(10, size = 1000, replace = TRUE))
mm = glmer(Y~X1+X2+X3+(1|site), data = df, family = binomial())
rr = partR2::partR2(mm, partvars = c("X1", "X2", "X3"), R2_type="conditional")
rr$BW

library(dominanceanalysis)
dominanceAnalysis(m)


YY = binomial()$linkfun(scales::rescale(as.numeric(df$Y), to = c(0.001, 1-0.001) ))
coef(m)

imp2 = function(df) {
  model =  glm(Y~X1+X2+X3, data= df, family = binomial()) 
  res = rep(0, 3)
  for(i in 1:3){ 
    model_tmp = model
    model_tmp$coefficients[2:4] = model_tmp$coefficients[2:4] * diag(1.0, 3)[i,]
    pred = predict(model_tmp, newdata = df)
    res[i] = 1-sum((YY - pred  )**2)/sum((YY)**2)
  }
  return(res)
}
imp2(df)

library(iml)
Pred = Predictor$new(model = m, data = df[,-c(1, 5)])
Pred$task = "classif"
res = sapply(1:1000, function(i) iml::Shapley$new(Pred, x.interest = df[i,2:4])$results[,2])


imp_glm2 = function(df) {
  model =  glm(Y~X1+X2+X3, family = binomial(), data = df)
  null = glm(Y~1, family = binomial(), data = df)
  pred_null = predict(null, newdata = df, type = "response")
  f_L = sum(dbinom(df$Y, 1, pred_null, TRUE))
  
  res = rep(0, 3)
  for(i in 1:3){ 
    df_tmp = df
    df_tmp[,i+1] = 0
    pred = predict(model, newdata = df_tmp, type = "response")
    f_m = sum(dbinom(df$Y, 1, pred, TRUE))
    res[i] = 1 -  f_m / f_L # pseudo Rsquared (McFadden)
  }
  res = c(res, 1- sum(dbinom(df$Y, 1, predict(model, newdata = df, type = "response"), TRUE))/f_L)
  res[1:3] = res[4] - res[1:3]
  
  return(res)
}
(imp_glm2(df))


library(Hmsc)

n = 200
X1 = runif(n, -1, 1)
X2 = 0*X1 + rnorm(n, sd = 0.3)
X3 = runif(n, -1 ,1 )
cor(X1, X2)
X1 = scale(X1)
X2 = scale(X2)
X3 = scale(X3)
prob = plogis(-2*X1 + 2*X3)
Y = rbinom(n, 1, prob)
df = data.frame(Y = Y, X1= X1, X2 = X2, X3 = X3)
m = glm(Y~X1+X2+X3, family = binomial(), data = df)
rsq::rsq.partial(m, type = "n")
car::Anova(glm(Y~X1+X2+X3, family = binomial(), data = df), error.estimate=c("deviance"), test.statistic = "F")
anova(m)
summary(m)
df = data.frame(Y = Y, X1= X1, X2 = X2, X3 = X3)


Y = as.matrix(df$Y)
XData = df[,-1]
m = Hmsc(Y = Y, XData = XData, XFormula = ~X1+X2+X3, distr = "normal")
nChains = 2
test.run = FALSE
if (test.run){
  #with this option, the vignette runs fast but results are not reliable
  thin = 1
  samples = 10
  transient = 5
  verbose = 5
} else {
  #with this option, the vignette evaluates slow but it reproduces the results of the
  #.pdf version
  thin = 5
  samples = 1000
  transient = 500*thin
  verbose = 500*thin
}

m = sampleMcmc(m, thin = thin, samples = samples, transient = transient,
               nChains = nChains, verbose = verbose)
Hmsc::computeVariancePartitioning(m)$vals



