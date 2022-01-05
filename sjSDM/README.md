
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
## ── Attaching sjSDM ──────────────────────────────────────────────────── 1.0.0 ──
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
## LogLik:  -575.5497 
## Regularization loss:  0 
## 
## Species-species correlation matrix: 
## 
##  sp1  1.0000                                 
##  sp2 -0.3430  1.0000                             
##  sp3 -0.0570 -0.3700  1.0000                         
##  sp4  0.0770 -0.3130  0.6730  1.0000                     
##  sp5  0.4870 -0.3630 -0.0450 -0.1260  1.0000                 
##  sp6 -0.0970  0.3320  0.1880  0.1880 -0.0880  1.0000             
##  sp7  0.4360 -0.2090  0.1800  0.1530  0.5090  0.2490  1.0000         
##  sp8  0.2350  0.0560 -0.3290 -0.3380  0.3190 -0.0090  0.2010  1.0000     
##  sp9  0.0400 -0.0330  0.0160  0.2050 -0.3410 -0.1920 -0.2980 -0.2220  1.0000 
##  sp10     0.3090  0.3230 -0.5050 -0.3800  0.1930  0.1360  0.1890  0.3570 -0.0930  1.0000
## 
## 
## 
## Spatial: 
##              sp1        sp2       sp3       sp4       sp5       sp6       sp7
## X1:X2 0.08259914 -0.5019723 0.4352706 -0.169348 0.4040644 0.2000494 0.4183032
##             sp8       sp9        sp10
## X1:X2 0.3042413 0.2288286 -0.02296865
## 
## 
## 
##                  Estimate  Std.Err Z value Pr(>|z|)    
## sp1 (Intercept)  -0.06842  0.20499   -0.33  0.73857    
## sp1 X1            0.72483  0.39751    1.82  0.06824 .  
## sp2 (Intercept)  -0.03794  0.20898   -0.18  0.85593    
## sp2 X1            0.92104  0.40582    2.27  0.02323 *  
## sp3 (Intercept)  -0.26596  0.21984   -1.21  0.22637    
## sp3 X1            0.95088  0.41292    2.30  0.02129 *  
## sp4 (Intercept)  -0.04988  0.20611   -0.24  0.80877    
## sp4 X1           -1.15879  0.38965   -2.97  0.00294 ** 
## sp5 (Intercept)  -0.15585  0.19463   -0.80  0.42326    
## sp5 X1            0.50131  0.37329    1.34  0.17929    
## sp6 (Intercept)   0.18499  0.19405    0.95  0.34043    
## sp6 X1            1.55482  0.39826    3.90  9.5e-05 ***
## sp7 (Intercept)   0.00426  0.17742    0.02  0.98084    
## sp7 X1           -0.34412  0.34556   -1.00  0.31933    
## sp8 (Intercept)   0.15873  0.13976    1.14  0.25607    
## sp8 X1            0.14996  0.26153    0.57  0.56637    
## sp9 (Intercept)   0.01891  0.17133    0.11  0.91211    
## sp9 X1            1.09219  0.32656    3.34  0.00082 ***
## sp10 (Intercept) -0.07258  0.17739   -0.41  0.68243    
## sp10 X1          -0.49947  0.32436   -1.54  0.12359    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

Update model (change main effects to quadratic effects):

``` r
model2 <- update(model, env_formula = ~I(X1^2))
summary(model2)
## LogLik:  -608.1784 
## Regularization loss:  0 
## 
## Species-species correlation matrix: 
## 
##  sp1  1.0000                                 
##  sp2 -0.3030  1.0000                             
##  sp3 -0.0130 -0.2560  1.0000                         
##  sp4  0.0600 -0.3530  0.5650  1.0000                     
##  sp5  0.4550 -0.3540 -0.0540 -0.0690  1.0000                 
##  sp6 -0.0420  0.4110  0.2420  0.0420 -0.0790  1.0000             
##  sp7  0.4220 -0.2130  0.1750  0.0780  0.5140  0.2090  1.0000         
##  sp8  0.2440  0.0710 -0.3000 -0.3740  0.3210  0.0220  0.2360  1.0000     
##  sp9  0.0920  0.0950  0.1040  0.1220 -0.3260  0.0200 -0.2400 -0.1540  1.0000 
##  sp10     0.2760  0.2130 -0.5060 -0.3470  0.2340  0.0230  0.1450  0.3220 -0.1210  1.0000
## 
## 
## 
## Spatial: 
##             sp1        sp2       sp3        sp4       sp5       sp6      sp7
## X1:X2 0.1553214 -0.4860287 0.4846192 -0.2180294 0.4662001 0.1784182 0.378258
##             sp8       sp9        sp10
## X1:X2 0.2950488 0.2057981 -0.03424472
## 
## 
## 
##                  Estimate Std.Err Z value Pr(>|z|)
## sp1 (Intercept)   -0.1303  0.2849   -0.46     0.65
## sp1 I(X1^2)        0.4434  0.6996    0.63     0.53
## sp2 (Intercept)   -0.2555  0.2864   -0.89     0.37
## sp2 I(X1^2)        0.7050  0.7215    0.98     0.33
## sp3 (Intercept)   -0.2640  0.2952   -0.89     0.37
## sp3 I(X1^2)        0.1099  0.6970    0.16     0.87
## sp4 (Intercept)   -0.2272  0.2737   -0.83     0.41
## sp4 I(X1^2)        0.7222  0.6841    1.06     0.29
## sp5 (Intercept)   -0.1383  0.2487   -0.56     0.58
## sp5 I(X1^2)        0.2325  0.6248    0.37     0.71
## sp6 (Intercept)    0.1669  0.2690    0.62     0.53
## sp6 I(X1^2)       -0.0286  0.6517   -0.04     0.97
## sp7 (Intercept)    0.0521  0.2466    0.21     0.83
## sp7 I(X1^2)       -0.0193  0.6236   -0.03     0.98
## sp8 (Intercept)   -0.0328  0.2086   -0.16     0.88
## sp8 I(X1^2)        0.7716  0.5249    1.47     0.14
## sp9 (Intercept)    0.2076  0.2478    0.84     0.40
## sp9 I(X1^2)       -0.6296  0.5992   -1.05     0.29
## sp10 (Intercept)  -0.1308  0.2505   -0.52     0.60
## sp10 I(X1^2)       0.2070  0.5999    0.35     0.73
```

Change linear part of model to a deep neural network:

``` r
DNN <- sjSDM(Y = Occ, env = DNN(data = Env, formula = ~.), spatial = linear(data = SP, formula = ~0+X1:X2), se = TRUE, family=binomial("probit"), sampling = 100L)
summary(DNN)
## LogLik:  -564.1605 
## Regularization loss:  0 
## 
## Species-species correlation matrix: 
## 
##  sp1  1.0000                                 
##  sp2 -0.4150  1.0000                             
##  sp3 -0.0480 -0.3220  1.0000                         
##  sp4  0.0440 -0.3230  0.5590  1.0000                     
##  sp5  0.5510 -0.2900 -0.0280 -0.0880  1.0000                 
##  sp6 -0.1760  0.3670  0.2260  0.0560  0.0130  1.0000             
##  sp7  0.5050 -0.1590  0.2130  0.0860  0.5020  0.3070  1.0000         
##  sp8  0.2690  0.1100 -0.2800 -0.3930  0.2870  0.0620  0.2770  1.0000     
##  sp9 -0.0630 -0.1070  0.0550  0.1380 -0.2600 -0.2420 -0.2260 -0.1470  1.0000 
##  sp10     0.2720  0.2690 -0.5240 -0.3000  0.2340  0.0440  0.1590  0.2880 -0.1800  1.0000
## 
## 
## 
## Spatial: 
##              sp1        sp2       sp3        sp4       sp5       sp6       sp7
## X1:X2 0.09318528 -0.4935189 0.4328462 -0.1994821 0.4871828 0.1590104 0.3766337
##             sp8       sp9         sp10
## X1:X2 0.2689047 0.2027397 -0.002816148
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
