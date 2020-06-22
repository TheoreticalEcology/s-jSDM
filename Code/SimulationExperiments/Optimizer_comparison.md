---
title: "Optimizer"
author: "Maximilian Pichler"
date: "15 6 2020"
output: 
  html_document: 
    keep_md: yes
    toc: true
---




```r
set.seed(42)
library(sjSDM)
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
communities = lapply(c(25, 50, 100), function(sp) lapply(1:10, function(i) simulate_SDM(env = 3L, sites = 100L, species = sp)))

optims = c("adamax", "rmsprop", "accsgd", "adabound", "sgd", "diffgrad")
scheduler = c(TRUE, FALSE)
lr = c(0.01, 0.005, 0.002, 0.001)
epochs = c(100, 200)
bs = c(5, 10, 30)
sp = c(25, 50, 100)
iter = 1:10


test = data.frame(expand.grid(optims, scheduler, lr, epochs, bs, sp, iter))
colnames(test) = c("optim", "scheduler","lr","epochs", "bs", "sp", "iter")
test$acc = NA
test$grads = NA
test$rmse = NA
test = test[order(test$sp),]
```



```r
n = 12
cuts = cut(1:nrow(test), breaks = n)
levels(cuts) = 1:n

tests = lapply(1:n, function(i) test[cuts==paste0(i),])

library(snow)
cl = snow::makeCluster(n)
nodes = unlist(snow::clusterEvalQ(cl, paste(Sys.info()[['nodename']], Sys.getpid(), sep='-')))
control = snow::clusterEvalQ(cl, {library(sjSDM)})
snow::clusterExport(cl, list("communities", "test", "tests", "nodes"))

results = parLapply(cl, 1:n, function(n) {
  tmp = tests[[n]]
  for(i in 1:nrow(tmp)) {
    if(tmp[i,]$sp == 25) j = 1
    else if(tmp[i,]$sp == 50) j = 2
    else j = 3
    com = communities[[j]][[tmp[i,]$iter]]
    
    opt = 
    switch (as.character(tmp[i,]$optim),
      adamax = Adamax(),
      rmsprop = RMSprop(),
      accsgd = AccSGD(),
      adabound = AdaBound(),
      sgd = SGD(),
      diffgrad = DiffGrad()
    )
    
    if(n %in% 1:4) device = 0
    else if(n %in% 5:8) device = 1
    else if(n %in% 9:12) device = 2
    
    
    m = sjSDM(com$response, com$env_weights, 
              learning_rate = tmp[i,]$lr, 
              step_size = tmp[i,]$bs, 
              iter = tmp[i,]$epochs, 
              control = sjSDMControl(opt, tmp[i,]$scheduler),
              device = device)
    
    tmp[i,]$acc = com$corr_acc(getCov(m))
    tmp[i,]$grads = m$model$params[[2]][[1]]$grad$pow(2.0)$mean()$sqrt()$item()
    tmp[i,]$rmse = sqrt(mean( ( t(coef(m)[[1]])[-1,] - com$species_weights )^2 ))
    rm(m)
  }
  return(tmp)
  
})
```


```r
res = do.call(rbind, results)
summary(lm(acc~as.factor(sp) + lr + epochs + optim, data = res))
```

```
## 
## Call:
## lm(formula = acc ~ as.factor(sp) + lr + epochs + optim, data = res)
## 
## Residuals:
##       Min        1Q    Median        3Q       Max 
## -0.265441 -0.019107  0.005541  0.025996  0.110204 
## 
## Coefficients:
##                    Estimate Std. Error t value Pr(>|t|)    
## (Intercept)       6.900e-01  1.926e-03 358.295   <2e-16 ***
## as.factor(sp)50  -4.762e-02  1.092e-03 -43.602   <2e-16 ***
## as.factor(sp)100 -1.071e-01  1.092e-03 -98.087   <2e-16 ***
## lr                5.255e+00  1.274e-01  41.253   <2e-16 ***
## epochs            1.020e-04  8.918e-06  11.443   <2e-16 ***
## optimrmsprop      3.945e-02  1.545e-03  25.541   <2e-16 ***
## optimaccsgd       3.176e-02  1.545e-03  20.560   <2e-16 ***
## optimadabound     4.048e-02  1.545e-03  26.209   <2e-16 ***
## optimsgd          1.439e-02  1.545e-03   9.318   <2e-16 ***
## optimdiffgrad     1.426e-02  1.545e-03   9.235   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.04145 on 8630 degrees of freedom
## Multiple R-squared:  0.5937,	Adjusted R-squared:  0.5933 
## F-statistic:  1401 on 9 and 8630 DF,  p-value: < 2.2e-16
```

```r
res2 = 
  res %>% 
    group_by(optim, scheduler, lr, epochs, bs, sp) %>% 
    summarise(acc = mean(acc), grads = mean(grads), rmse = mean(rmse))
```

```
## Warning: The `printer` argument is deprecated as of rlang 0.3.0.
## This warning is displayed once per session.
```


## Generally

```r
par(mfrow=c(1,2))
boxplot(res2$acc~res2$optim, ylim = c(0.5, 1.0))
boxplot(res2$rmse~res2$optim, ylim = c(0.0, 0.6))
```

![](/home/maxpichler/sjSDM/deepJSDM/Code/SimulationExperiments/Optimizer_comparison_files/figure-html/unnamed-chunk-4-1.png)<!-- -->
RMSprop and AdaBound are the most stable, Adamax the worst


## Learning rate

```r
par(mfrow=c(1,3))
for(i in c(0.01, 0.005, 0.001)) boxplot(res2$acc[res2$lr==i]~res2$optim[res2$lr==i], ylim = c(0.5, 1.0))
```

![](/home/maxpichler/sjSDM/deepJSDM/Code/SimulationExperiments/Optimizer_comparison_files/figure-html/unnamed-chunk-5-1.png)<!-- -->
AdaBound is influenced the least by the learning rate!


## Epochs

```r
par(mfrow=c(1,2))
for(i in c(100,200)) boxplot(res2$acc[res2$epochs==i]~res2$optim[res2$epochs==i], ylim = c(0.5, 1.0))
```

![](/home/maxpichler/sjSDM/deepJSDM/Code/SimulationExperiments/Optimizer_comparison_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

## Batch size

```r
par(mfrow=c(1,3))
for(i in bs) boxplot(res2$acc[res2$bs==i]~res2$optim[res2$bs==i], ylim = c(0.5, 1.0))
```

![](/home/maxpichler/sjSDM/deepJSDM/Code/SimulationExperiments/Optimizer_comparison_files/figure-html/unnamed-chunk-7-1.png)<!-- -->
RMSprop and AdaBound are insensitive against the batch size


## With or without scheduler:

```r
par(mfrow=c(1,2))
boxplot(res2$acc[ !res2$scheduler]~res2$optim[!res2$scheduler], ylim = c(0.5, 1.0))
boxplot(res2$acc[ res2$scheduler]~res2$optim[res2$scheduler], ylim = c(0.5, 1.0))
```

![](/home/maxpichler/sjSDM/deepJSDM/Code/SimulationExperiments/Optimizer_comparison_files/figure-html/unnamed-chunk-8-1.png)<!-- -->


## RMSprop vs AdaBound

```r
summary(lm(acc~., data=res[res$optim=="rmsprop",-c(1,9,10)]))
```

```
## 
## Call:
## lm(formula = acc ~ ., data = res[res$optim == "rmsprop", -c(1, 
##     9, 10)])
## 
## Residuals:
##       Min        1Q    Median        3Q       Max 
## -0.120138 -0.013056  0.000646  0.014457  0.074243 
## 
## Coefficients:
##                 Estimate Std. Error t value Pr(>|t|)    
## (Intercept)    7.978e-01  3.073e-03 259.594  < 2e-16 ***
## schedulerTRUE -4.904e-03  1.318e-03  -3.721 0.000206 ***
## lr             1.780e+00  1.883e-01   9.454  < 2e-16 ***
## epochs         3.989e-05  1.318e-05   3.027 0.002518 ** 
## bs            -6.489e-04  6.101e-05 -10.636  < 2e-16 ***
## sp            -1.484e-03  2.114e-05 -70.194  < 2e-16 ***
## iter           6.786e-04  2.294e-04   2.958 0.003151 ** 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.02501 on 1433 degrees of freedom
## Multiple R-squared:  0.7827,	Adjusted R-squared:  0.7818 
## F-statistic: 860.2 on 6 and 1433 DF,  p-value: < 2.2e-16
```

```r
summary(lm(acc~., data=res[res$optim=="adabound",-c(1,9,10)]))
```

```
## 
## Call:
## lm(formula = acc ~ ., data = res[res$optim == "adabound", -c(1, 
##     9, 10)])
## 
## Residuals:
##       Min        1Q    Median        3Q       Max 
## -0.074650 -0.013398  0.000223  0.013675  0.073918 
## 
## Coefficients:
##                 Estimate Std. Error t value Pr(>|t|)    
## (Intercept)    7.953e-01  2.699e-03 294.635  < 2e-16 ***
## schedulerTRUE  5.363e-03  1.158e-03   4.633 3.93e-06 ***
## lr             5.552e-01  1.654e-01   3.357 0.000807 ***
## epochs         1.965e-05  1.158e-05   1.698 0.089792 .  
## bs            -9.121e-05  5.358e-05  -1.702 0.088947 .  
## sp            -1.502e-03  1.856e-05 -80.892  < 2e-16 ***
## iter           6.202e-04  2.015e-04   3.078 0.002126 ** 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.02196 on 1433 degrees of freedom
## Multiple R-squared:  0.8214,	Adjusted R-squared:  0.8207 
## F-statistic:  1099 on 6 and 1433 DF,  p-value: < 2.2e-16
```

```r
summary(lm(acc~., data=res[res$optim=="adamax",-c(1,9,10)]))
```

```
## 
## Call:
## lm(formula = acc ~ ., data = res[res$optim == "adamax", -c(1, 
##     9, 10)])
## 
## Residuals:
##       Min        1Q    Median        3Q       Max 
## -0.184134 -0.026795  0.004025  0.028950  0.114913 
## 
## Coefficients:
##                 Estimate Std. Error t value Pr(>|t|)    
## (Intercept)    7.083e-01  5.244e-03 135.070   <2e-16 ***
## schedulerTRUE -1.967e-02  2.249e-03  -8.748   <2e-16 ***
## lr             1.097e+01  3.213e-01  34.158   <2e-16 ***
## epochs         2.046e-04  2.249e-05   9.101   <2e-16 ***
## bs            -2.217e-03  1.041e-04 -21.296   <2e-16 ***
## sp            -1.240e-03  3.606e-05 -34.398   <2e-16 ***
## iter           8.104e-04  3.915e-04   2.070   0.0386 *  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.04267 on 1433 degrees of freedom
## Multiple R-squared:  0.6743,	Adjusted R-squared:  0.673 
## F-statistic: 494.5 on 6 and 1433 DF,  p-value: < 2.2e-16
```

