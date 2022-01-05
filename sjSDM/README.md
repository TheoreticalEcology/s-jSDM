
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

The method is described in Pichler & Hartig (2021)
\<doi.org/10.1111/2041-210X.13687> A new joint species distribution
model for faster and more accurate inference of species associations
from big community data.

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
## ── Attaching sjSDM ──────────────────────────────────────────────────── 0.1.9 ──
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
## LogLik:  -580.8655 
## Regularization loss:  0 
## 
## Species-species correlation matrix: 
## 
##  sp1  1.0000                                 
##  sp2 -0.3680  1.0000                             
##  sp3 -0.0250 -0.3610  1.0000                         
##  sp4  0.1130 -0.2970  0.6650  1.0000                     
##  sp5  0.4580 -0.3710 -0.0040 -0.0970  1.0000                 
##  sp6 -0.1220  0.3450  0.1980  0.2110 -0.0720  1.0000             
##  sp7  0.4210 -0.2160  0.1910  0.1620  0.5220  0.2370  1.0000         
##  sp8  0.2310  0.0560 -0.3160 -0.3220  0.3320 -0.0100  0.2310  1.0000     
##  sp9  0.0460 -0.0230 -0.0030  0.1760 -0.3760 -0.1890 -0.3110 -0.2140  1.0000 
##  sp10     0.3070  0.2990 -0.4890 -0.3420  0.1850  0.1120  0.1850  0.3550 -0.0720  1.0000
## 
## 
## 
## Spatial: 
##             sp1        sp2       sp3        sp4       sp5       sp6       sp7
## X1:X2 0.1225304 -0.4919888 0.3932282 -0.1778442 0.4074326 0.1737909 0.3971201
##             sp8       sp9        sp10
## X1:X2 0.3096057 0.2311323 -0.01854839
## 
## 
## 
##                  Estimate Std.Err Z value Pr(>|z|)    
## sp1 (Intercept)   -0.0276  0.2099   -0.13  0.89518    
## sp1 X1             0.6783  0.4044    1.68  0.09349 .  
## sp2 (Intercept)   -0.0494  0.2074   -0.24  0.81189    
## sp2 X1             0.9602  0.4019    2.39  0.01689 *  
## sp3 (Intercept)   -0.2717  0.2090   -1.30  0.19357    
## sp3 X1             0.9571  0.4115    2.33  0.02003 *  
## sp4 (Intercept)   -0.0477  0.2126   -0.22  0.82249    
## sp4 X1            -1.1682  0.4109   -2.84  0.00447 ** 
## sp5 (Intercept)   -0.1410  0.1945   -0.72  0.46856    
## sp5 X1             0.4416  0.3633    1.22  0.22423    
## sp6 (Intercept)    0.1875  0.1918    0.98  0.32831    
## sp6 X1             1.5792  0.3963    3.99  6.7e-05 ***
## sp7 (Intercept)    0.0311  0.1767    0.18  0.86018    
## sp7 X1            -0.3637  0.3427   -1.06  0.28849    
## sp8 (Intercept)    0.1668  0.1403    1.19  0.23452    
## sp8 X1             0.1424  0.2602    0.55  0.58408    
## sp9 (Intercept)    0.0209  0.1783    0.12  0.90664    
## sp9 X1             1.1015  0.3335    3.30  0.00096 ***
## sp10 (Intercept)  -0.0681  0.1784   -0.38  0.70277    
## sp10 X1           -0.5161  0.3252   -1.59  0.11254    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

Update model (change main effects to quadratic effects):

``` r
model2 <- update(model, env_formula = ~I(X1^2))
summary(model2)
## LogLik:  -607.8209 
## Regularization loss:  0 
## 
## Species-species correlation matrix: 
## 
##  sp1  1.0000                                 
##  sp2 -0.2930  1.0000                             
##  sp3  0.0020 -0.2810  1.0000                         
##  sp4  0.0730 -0.3520  0.5630  1.0000                     
##  sp5  0.4390 -0.3220 -0.0400 -0.0650  1.0000                 
##  sp6 -0.0600  0.4210  0.2250  0.0400 -0.0660  1.0000             
##  sp7  0.4160 -0.2050  0.1940  0.0840  0.5100  0.2050  1.0000         
##  sp8  0.2380  0.0430 -0.2770 -0.4020  0.3240 -0.0090  0.2350  1.0000     
##  sp9  0.0960  0.1000  0.0930  0.1260 -0.3180  0.0210 -0.2380 -0.1470  1.0000 
##  sp10     0.2690  0.2360 -0.4970 -0.3420  0.2380  0.0330  0.1430  0.2800 -0.1020  1.0000
## 
## 
## 
## Spatial: 
##            sp1        sp2       sp3        sp4       sp5       sp6       sp7
## X1:X2 0.159807 -0.4460613 0.4789282 -0.2182595 0.4713256 0.1966043 0.3753874
##             sp8       sp9         sp10
## X1:X2 0.2649812 0.2294899 -0.003077535
## 
## 
## 
##                  Estimate Std.Err Z value Pr(>|z|)
## sp1 (Intercept)   -0.1697  0.2762   -0.61     0.54
## sp1 I(X1^2)        0.4934  0.6979    0.71     0.48
## sp2 (Intercept)   -0.1864  0.3003   -0.62     0.53
## sp2 I(X1^2)        0.6698  0.7665    0.87     0.38
## sp3 (Intercept)   -0.2764  0.3015   -0.92     0.36
## sp3 I(X1^2)        0.0904  0.7337    0.12     0.90
## sp4 (Intercept)   -0.2721  0.2814   -0.97     0.33
## sp4 I(X1^2)        0.7587  0.6896    1.10     0.27
## sp5 (Intercept)   -0.1547  0.2428   -0.64     0.52
## sp5 I(X1^2)        0.2422  0.6151    0.39     0.69
## sp6 (Intercept)    0.2049  0.2730    0.75     0.45
## sp6 I(X1^2)       -0.0604  0.6746   -0.09     0.93
## sp7 (Intercept)    0.0272  0.2484    0.11     0.91
## sp7 I(X1^2)       -0.0224  0.6175   -0.04     0.97
## sp8 (Intercept)   -0.0231  0.2121   -0.11     0.91
## sp8 I(X1^2)        0.7838  0.5439    1.44     0.15
## sp9 (Intercept)    0.1963  0.2392    0.82     0.41
## sp9 I(X1^2)       -0.6173  0.5996   -1.03     0.30
## sp10 (Intercept)  -0.1286  0.2434   -0.53     0.60
## sp10 I(X1^2)       0.2200  0.5864    0.38     0.71
```

Change linear part of model to a deep neural network:

``` r
DNN <- sjSDM(Y = Occ, env = DNN(data = Env, formula = ~.), spatial = linear(data = SP, formula = ~0+X1:X2), se = TRUE, family=binomial("probit"), sampling = 100L)
summary(DNN)
## LogLik:  -615.3118 
## Regularization loss:  0 
## 
## Species-species correlation matrix: 
## 
##  sp1  1.0000                                 
##  sp2 -0.2770  1.0000                             
##  sp3 -0.0270 -0.2710  1.0000                         
##  sp4  0.0550 -0.3530  0.5470  1.0000                     
##  sp5  0.4690 -0.3110 -0.0420 -0.0380  1.0000                 
##  sp6 -0.0410  0.4150  0.2450  0.0430 -0.0560  1.0000             
##  sp7  0.4240 -0.1760  0.1670  0.0980  0.5170  0.2090  1.0000         
##  sp8  0.2500  0.0660 -0.2750 -0.3630  0.3340  0.0220  0.2450  1.0000     
##  sp9  0.0970  0.0660  0.0690  0.0760 -0.3090  0.0030 -0.2430 -0.1540  1.0000 
##  sp10     0.2880  0.2490 -0.5000 -0.3300  0.2640  0.0360  0.1840  0.3130 -0.1280  1.0000
## 
## 
## 
## Spatial: 
##             sp1       sp2       sp3        sp4      sp5     sp6       sp7
## X1:X2 0.1598141 -0.488793 0.4645391 -0.2129406 0.488393 0.17316 0.4118766
##             sp8       sp9        sp10
## X1:X2 0.2884549 0.2118452 -0.00479748
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
