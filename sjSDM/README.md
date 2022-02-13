
<!-- README.md is generated from README.Rmd. Please edit that file -->

# sjSDM - Fast and accurate Joint Species Distribution Modeling

[![License: GPL
v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![R-CMD-check](https://github.com/TheoreticalEcology/s-jSDM/workflows/R-CMD-check/badge.svg?branch=master)
[![Publication](https://img.shields.io/badge/Publication-10.1111/2041-green.svg)](https://besjournals.onlinelibrary.wiley.com/doi/abs/10.1111/2041-210X.13687)

## Overview

A scalable method to estimates joint Species Distribution Models (jSDMs)
based on the multivariate probit model through Monte-Carlo approximation
of the joint likelihood. The numerical approximation is based on
‘PyTorch’ and ‘reticulate’, and can be calculated on CPUs and GPUs
alike.

The method is described in [Pichler & Hartig
(2021)](https://besjournals.onlinelibrary.wiley.com/doi/abs/10.1111/2041-210X.13687)
A new joint species distribution model for faster and more accurate
inference of species associations from big community data.

The package includes options to fit various different (j)SDM models:

-   jSDMs with Binomial, Poisson and Normal distributed responses
-   jSDMs based on deep neural networks
-   Spatial auto-correlation can be accounted for by spatial
    eigenvectors or trend surface polynomials

To get more information, install the package and run

``` r
library(sjSDM)
?sjSDM
vignette("sjSDM", package="sjSDM")
```

## Installation

**sjSDM** is based on ‘PyTorch’, a ‘python’ library, and thus requires
‘python’ dependencies. The ‘python’ dependencies can be automatically
installed by running:

``` r
library(sjSDM)
install_sjSDM()
```

If this didn’t work, please check the troubleshooting guide:

``` r
library(sjSDM)
?installation_help
```

## Usage

Let’s first simulate a community data set:

``` r
library(sjSDM)
## ── Attaching sjSDM ──────────────────────────────────────────────────── 1.0.1 ──
## ✓ torch <environment> 
## ✓ torch_optimizer  
## ✓ pyro  
## ✓ madgrad
set.seed(42)
community <- simulate_SDM(sites = 100, species = 10, env = 3, se = TRUE)
Env <- community$env_weights
Occ <- community$response
SP <- matrix(rnorm(200, 0, 0.3), 100, 2) # spatial coordinates (no effect on species occurences)
```

Estimate jSDM:

``` r
model <- sjSDM(Y = Occ, env = linear(data = Env, formula = ~X1), spatial = linear(data = SP, formula = ~0+X1:X2), se = TRUE, family=binomial("probit"), sampling = 100L)
summary(model)
## LogLik:  -573.5576 
## Regularization loss:  0 
## 
## Species-species correlation matrix: 
## 
##  sp1  1.0000                                 
##  sp2 -0.3350  1.0000                             
##  sp3 -0.0490 -0.3660  1.0000                         
##  sp4  0.1080 -0.2780  0.6650  1.0000                     
##  sp5  0.4840 -0.3860 -0.0210 -0.1020  1.0000                 
##  sp6 -0.1210  0.3520  0.1870  0.2170 -0.1050  1.0000             
##  sp7  0.4520 -0.2210  0.1640  0.1610  0.5270  0.2060  1.0000         
##  sp8  0.2240  0.0570 -0.3170 -0.3180  0.3270 -0.0120  0.2270  1.0000     
##  sp9  0.0280  0.0030 -0.0190  0.1620 -0.3620 -0.1890 -0.3090 -0.2150  1.0000 
##  sp10     0.3060  0.3170 -0.4950 -0.3340  0.1960  0.1290  0.2080  0.3440 -0.0880  1.0000
## 
## 
## 
## Spatial: 
##             sp1        sp2       sp3        sp4       sp5       sp6       sp7
## X1:X2 0.1443628 -0.4960446 0.3901674 -0.1664559 0.3926543 0.1800613 0.4187593
##             sp8       sp9        sp10
## X1:X2 0.3037927 0.2392421 -0.02216352
## 
## 
## 
##                  Estimate  Std.Err Z value Pr(>|z|)    
## sp1 (Intercept)  -0.02966  0.20741   -0.14  0.88629    
## sp1 X1            0.71848  0.40269    1.78  0.07439 .  
## sp2 (Intercept)  -0.03965  0.21109   -0.19  0.85101    
## sp2 X1            0.88713  0.40121    2.21  0.02703 *  
## sp3 (Intercept)  -0.28850  0.21880   -1.32  0.18731    
## sp3 X1            1.00122  0.41875    2.39  0.01681 *  
## sp4 (Intercept)  -0.06132  0.21224   -0.29  0.77265    
## sp4 X1           -1.14096  0.40993   -2.78  0.00538 ** 
## sp5 (Intercept)  -0.12334  0.19810   -0.62  0.53356    
## sp5 X1            0.50015  0.37202    1.34  0.17881    
## sp6 (Intercept)   0.18113  0.18482    0.98  0.32708    
## sp6 X1            1.59964  0.37527    4.26    2e-05 ***
## sp7 (Intercept)   0.04020  0.17696    0.23  0.82029    
## sp7 X1           -0.31613  0.33246   -0.95  0.34166    
## sp8 (Intercept)   0.17521  0.13905    1.26  0.20764    
## sp8 X1            0.14439  0.25896    0.56  0.57714    
## sp9 (Intercept)   0.00971  0.16985    0.06  0.95440    
## sp9 X1            1.08921  0.32526    3.35  0.00081 ***
## sp10 (Intercept) -0.04666  0.17873   -0.26  0.79404    
## sp10 X1          -0.51466  0.32894   -1.56  0.11767    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

Update model (change main effects to quadratic effects):

``` r
model2 <- update(model, env_formula = ~I(X1^2))
summary(model2)
## LogLik:  -600.7468 
## Regularization loss:  0 
## 
## Species-species correlation matrix: 
## 
##  sp1  1.0000                                 
##  sp2 -0.2790  1.0000                             
##  sp3 -0.0050 -0.2990  1.0000                         
##  sp4  0.0740 -0.3740  0.5660  1.0000                     
##  sp5  0.4610 -0.3080 -0.0470 -0.0500  1.0000                 
##  sp6 -0.0200  0.4190  0.2160  0.0220 -0.0430  1.0000             
##  sp7  0.4240 -0.1900  0.1680  0.0770  0.5220  0.2100  1.0000         
##  sp8  0.2310  0.0920 -0.2860 -0.3750  0.3000  0.0370  0.2250  1.0000     
##  sp9  0.0620  0.0730  0.0930  0.1290 -0.3350  0.0040 -0.2590 -0.1520  1.0000 
##  sp10     0.2660  0.2520 -0.5050 -0.3390  0.2540  0.0410  0.1520  0.2880 -0.1350  1.0000
## 
## 
## 
## Spatial: 
##             sp1        sp2       sp3        sp4       sp5       sp6       sp7
## X1:X2 0.1792488 -0.4773207 0.4441337 -0.2305051 0.5176902 0.1653531 0.3966912
##             sp8      sp9        sp10
## X1:X2 0.3105257 0.205059 0.001748362
## 
## 
## 
##                  Estimate Std.Err Z value Pr(>|z|)
## sp1 (Intercept)   -0.1326  0.2967   -0.45     0.65
## sp1 I(X1^2)        0.4920  0.7335    0.67     0.50
## sp2 (Intercept)   -0.1991  0.3107   -0.64     0.52
## sp2 I(X1^2)        0.6705  0.7686    0.87     0.38
## sp3 (Intercept)   -0.3092  0.3054   -1.01     0.31
## sp3 I(X1^2)        0.0981  0.7141    0.14     0.89
## sp4 (Intercept)   -0.2525  0.2741   -0.92     0.36
## sp4 I(X1^2)        0.7372  0.6466    1.14     0.25
## sp5 (Intercept)   -0.1684  0.2489   -0.68     0.50
## sp5 I(X1^2)        0.2164  0.6206    0.35     0.73
## sp6 (Intercept)    0.1737  0.2726    0.64     0.52
## sp6 I(X1^2)       -0.0424  0.6614   -0.06     0.95
## sp7 (Intercept)    0.0258  0.2315    0.11     0.91
## sp7 I(X1^2)       -0.0231  0.6114   -0.04     0.97
## sp8 (Intercept)   -0.0281  0.2060   -0.14     0.89
## sp8 I(X1^2)        0.7596  0.5269    1.44     0.15
## sp9 (Intercept)    0.2078  0.2358    0.88     0.38
## sp9 I(X1^2)       -0.6379  0.5750   -1.11     0.27
## sp10 (Intercept)  -0.0949  0.2488   -0.38     0.70
## sp10 I(X1^2)       0.2110  0.5790    0.36     0.72
```

Change linear part of model to a deep neural network:

``` r
DNN <- sjSDM(Y = Occ, env = DNN(data = Env, formula = ~.), spatial = linear(data = SP, formula = ~0+X1:X2), se = TRUE, family=binomial("probit"), sampling = 100L)
summary(DNN)
## LogLik:  -614.9956 
## Regularization loss:  0 
## 
## Species-species correlation matrix: 
## 
##  sp1  1.0000                                 
##  sp2 -0.2870  1.0000                             
##  sp3 -0.0150 -0.2860  1.0000                         
##  sp4  0.0800 -0.3620  0.5560  1.0000                     
##  sp5  0.4460 -0.3200 -0.0600 -0.0580  1.0000                 
##  sp6 -0.0510  0.4210  0.2060  0.0210 -0.0580  1.0000             
##  sp7  0.4260 -0.2010  0.1570  0.0960  0.5190  0.1920  1.0000         
##  sp8  0.2390  0.0680 -0.2780 -0.3490  0.3270  0.0330  0.2370  1.0000     
##  sp9  0.0910  0.0900  0.0780  0.0870 -0.3270  0.0010 -0.2350 -0.1370  1.0000 
##  sp10     0.2690  0.2530 -0.5070 -0.3200  0.2500  0.0530  0.1680  0.2980 -0.1210  1.0000
## 
## 
## 
## Spatial: 
##             sp1        sp2      sp3        sp4       sp5       sp6       sp7
## X1:X2 0.1575196 -0.4823389 0.471328 -0.2132361 0.4694341 0.2163433 0.3932205
##            sp8       sp9        sp10
## X1:X2 0.291459 0.2447751 -0.01574056
## 
## 
## 
## Env architecture:
## ===================================
## Layer_1:  (4, 10)
## Layer_2:  ReLU
## Layer_3:  (10, 10)
## Layer_4:  ReLU
## Layer_5:  (10, 10)
## Layer_6:  ReLU
## Layer_7:  (10, 10)
## ===================================
## Weights :     340
```
