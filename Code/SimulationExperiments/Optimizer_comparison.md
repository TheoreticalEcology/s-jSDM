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
## Warning: As of rlang 0.4.0, dplyr must be at least version 0.8.0.
## * dplyr 0.7.8 is too old for rlang 0.4.6.
## * Please update dplyr with `install.packages("dplyr")` and restart R.
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
communities = lapply(c(50, 100, 150), function(sp) lapply(1:10, function(i) simulate_SDM(env = 3L, sites = 200L, species = sp)))

optims = c("adamax", "rmsprop", "accsgd", "adabound", "sgd", "diffgrad")
scheduler = c(TRUE, FALSE)
lr = c(0.01, 0.005, 0.002, 0.001)
epochs = c(100, 200)
bs = c(10, 30, 50)
sp = c(50, 100, 150)
iter = 1:10


test = data.frame(expand.grid(optims, scheduler, lr, epochs, bs, sp, iter))
colnames(test) = c("optim", "scheduler","lr","epochs", "bs", "sp", "iter")
test$acc = NA
test$grads = NA
test = test[order(test$sp),]
```



```r
cuts = cut(1:nrow(test), breaks = 12)
levels(cuts) = 1:12

tests = lapply(1:12, function(i) test[cuts==paste0(i),])

library(snow)
cl = snow::makeCluster(12)
nodes = unlist(snow::clusterEvalQ(cl, paste(Sys.info()[['nodename']], Sys.getpid(), sep='-')))
control = snow::clusterEvalQ(cl, {library(sjSDM)})
snow::clusterExport(cl, list("communities", "test", "tests", "nodes"))

results = parLapply(cl, 1:12, function(n) {
  tmp = tests[[n]]
  for(i in 1:nrow(tmp)) {
    if(tmp[i,]$sp == 50) j = 1
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
    
    if(n %in% 1:3) device = 0
    else if(n %in% 4:8) device = 1
    else if(n %in% 9:12) device = 2
    
    
    m = sjSDM(com$response, com$env_weights, 
              learning_rate = tmp[i,]$lr, 
              step_size = tmp[i,]$bs, 
              iter = tmp[i,]$epochs, 
              control = sjSDMControl(opt, tmp[i,]$scheduler),
              device = device)
    
    tmp[i,]$acc = com$corr_acc(getCov(m))
    tmp[i,]$grads = m$model$params[[2]][[1]]$grad$pow(2.0)$mean()$sqrt()$item()
    rm(m)
  }
  return(tmp)
  
})
```


```r
res = do.call(rbind, results)
summary(lm(grads~as.factor(sp) + lr + epochs + optim, data = res))
```

```
## 
## Call:
## lm(formula = grads ~ as.factor(sp) + lr + epochs + optim, data = res)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -0.11195 -0.06355 -0.03403  0.07806  0.36792 
## 
## Coefficients:
##                    Estimate Std. Error t value Pr(>|t|)    
## (Intercept)       1.777e-01  3.873e-03  45.872  < 2e-16 ***
## as.factor(sp)100  4.955e-02  2.197e-03  22.558  < 2e-16 ***
## as.factor(sp)150  4.969e-02  2.197e-03  22.621  < 2e-16 ***
## lr                3.358e-01  2.562e-01   1.311   0.1901    
## epochs           -9.051e-06  1.794e-05  -0.505   0.6138    
## optimrmsprop     -2.588e-03  3.106e-03  -0.833   0.4048    
## optimaccsgd       6.436e-03  3.106e-03   2.072   0.0383 *  
## optimadabound     1.825e-02  3.106e-03   5.874 4.41e-09 ***
## optimsgd         -4.113e-04  3.106e-03  -0.132   0.8947    
## optimdiffgrad    -1.190e-03  3.106e-03  -0.383   0.7017    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.08336 on 8630 degrees of freedom
## Multiple R-squared:  0.07968,	Adjusted R-squared:  0.07872 
## F-statistic: 83.02 on 9 and 8630 DF,  p-value: < 2.2e-16
```

```r
res2 = 
  res %>% 
    group_by(optim, scheduler, lr, epochs, bs, sp) %>% 
    summarise(acc = mean(acc), grads = mean(grads))
```

```
## Warning: The `printer` argument is deprecated as of rlang 0.3.0.
## This warning is displayed once per session.
```


## Generally

```r
par(mfrow=c(1,1))
boxplot(res2$acc~res2$optim, ylim = c(0.5, 1.0))
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
for(i in c(10, 30, 50)) boxplot(res2$acc[res2$bs==i]~res2$optim[res2$bs==i], ylim = c(0.5, 1.0))
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

