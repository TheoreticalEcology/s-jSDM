---
title: "README"
author: "Max Pichler"
output: 
  html_document: 
    keep_md: yes
---



[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![R-CMD-check](https://github.com/TheoreticalEcology/s-jSDM/workflows/R-CMD-check/badge.svg?branch=master)


# s-jSDM - Fast and accurate Joint Species Distribution Modeling

## About the method

The method is described in the preprint Pichler & Hartig (2020) A new method for faster and more accurate inference of species associations from novel community data, https://arxiv.org/abs/2003.05331. The code for producing the results in this paper is available under the subfolder publications in this repo.

The method itself is wrapped into an R package, available under subfolder sjSDM. You can also use it stand-alone under Python (see instructions below). Note: for both the R and the python package, python >= 3.6 and pytorch must be installed (more details below).

## Installing the R / Python package

### R-package

Install the package via


```r
devtools::install_github("https://github.com/TheoreticalEcology/s-jSDM", subdir = "sjSDM")
```

Depencies for the package can be installed before or after installing the package. Detailed explanations of the dependencies are provided in vignette("Dependencies", package = "sjSDM"), source code [here](https://github.com/TheoreticalEcology/s-jSDM/blob/master/sjSDM/vignettes/Dependencies.Rmd). Very briefly,  the dependencies can be automatically installed from within R:


```r
sjSDM::install_sjSDM(version = "gpu") # or
sjSDM::install_sjSDM(version = "cpu")
```
Once the dependencies are installed, the following code should run:

Simulate a community and fit model:

```r
library(sjSDM)
community <- simulate_SDM(sites = 400, species = 10, env = 2)
Env <- community$env_weights
Occ <- community$response
SP <- matrix(rnorm(800), 400, 2) # spatial coordinates (no effect on species occurences)

model <- sjSDM(Y = Occ, env = linear(data = Env, formula = ~0+X1:X2 + X1), spatial = linear(data = SP, formula = ~0+X1:X2), se = TRUE)
summary(model)
```

```
## LogLik:  -2272.959 
## Deviance:  4545.919 
## 
## Regularization loss:  0 
## 
## Species-species correlation matrix: 
## 
##         sp1    sp2    sp3    sp4    sp5    sp6    sp7   sp8    sp9 sp10
## sp1   1.000  0.000  0.000  0.000  0.000  0.000  0.000 0.000  0.000    0
## sp2  -0.860  1.000  0.000  0.000  0.000  0.000  0.000 0.000  0.000    0
## sp3  -0.614  0.841  1.000  0.000  0.000  0.000  0.000 0.000  0.000    0
## sp4   0.634 -0.569 -0.539  1.000  0.000  0.000  0.000 0.000  0.000    0
## sp5   0.728 -0.724 -0.769  0.258  1.000  0.000  0.000 0.000  0.000    0
## sp6  -0.583  0.353  0.504 -0.582 -0.706  1.000  0.000 0.000  0.000    0
## sp7  -0.371  0.366  0.379  0.250 -0.831  0.449  1.000 0.000  0.000    0
## sp8   0.802 -0.660 -0.354  0.812  0.356 -0.494 -0.036 1.000  0.000    0
## sp9   0.267 -0.572 -0.769  0.396  0.266 -0.017  0.159 0.012  1.000    0
## sp10  0.899 -0.565 -0.354  0.657  0.589 -0.742 -0.268 0.793 -0.012    1
## 
## 
## 
## Spatial: 
##               sp1        sp2       sp3         sp4         sp5        sp6
## X1:X2 0.005453303 0.09658484 0.1103004 -0.03535753 -0.06153478 0.04092076
##              sp7         sp8        sp9         sp10
## X1:X2 0.07587598 0.008521888 -0.1270586 -0.007633902
## 
## 
## 
##            Estimate Std.Err Z value Pr(>|z|)    
## sp1 X1       0.1942  0.0858    2.26   0.0236 *  
## sp1 X1:X2    0.2644  0.1436    1.84   0.0656 .  
## sp2 X1      -0.0599  0.1380   -0.43   0.6644    
## sp2 X1:X2   -0.0512  0.2300   -0.22   0.8238    
## sp3 X1      -0.5393  0.1102   -4.89  9.9e-07 ***
## sp3 X1:X2   -0.0280  0.1860   -0.15   0.8804    
## sp4 X1      -0.6738  0.1151   -5.85  4.8e-09 ***
## sp4 X1:X2   -0.0797  0.1889   -0.42   0.6729    
## sp5 X1      -0.3478  0.0827   -4.21  2.6e-05 ***
## sp5 X1:X2    0.0979  0.1383    0.71   0.4791    
## sp6 X1       0.0450  0.1099    0.41   0.6822    
## sp6 X1:X2   -0.4545  0.1842   -2.47   0.0136 *  
## sp7 X1       0.8945  0.1453    6.15  7.5e-10 ***
## sp7 X1:X2   -0.1045  0.2386   -0.44   0.6615    
## sp8 X1      -0.2941  0.1070   -2.75   0.0060 ** 
## sp8 X1:X2   -0.1220  0.1809   -0.67   0.5001    
## sp9 X1      -0.1014  0.1177   -0.86   0.3892    
## sp9 X1:X2    0.0260  0.2048    0.13   0.8988    
## sp10 X1      0.3292  0.0982    3.35   0.0008 ***
## sp10 X1:X2   0.0900  0.1645    0.55   0.5842    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```
Let's have a look at the importance of the three groups (environment, associations, and space) on the occurences:

```r
imp = importance(model)
print(imp)
```

```
##    sp          env      spatial    biotic
## 1   1 0.0229018253 4.552773e-05 0.9770526
## 2   2 0.0003300774 2.749143e-03 0.9969208
## 3   3 0.0584046309 8.569510e-03 0.9330259
## 4   4 0.0920947823 8.979327e-04 0.9070073
## 5   5 0.0933394082 9.370771e-03 0.8972898
## 6   6 0.0299366964 2.222211e-03 0.9678411
## 7   7 0.1716630418 4.176085e-03 0.8241609
## 8   8 0.0220806251 6.588372e-05 0.9778535
## 9   9 0.0039590281 2.012255e-02 0.9759184
## 10 10 0.0366910423 7.055668e-05 0.9632384
```

```r
plot(imp)
```

![](README_files/figure-html/unnamed-chunk-4-1.png)<!-- -->


As expected, space has no effect on occurences.

Let's have a look on community level how the three groups contribute to the overall explained variance 

```r
an = anova(model, cv = FALSE)
print(an)
```

```
## Changes relative to empty model (without modules):
## 
##  Modules      LogLik            R2   marginal R2
##        _ 2772.587646  0.000000e+00  0.000000e+00
##        A  -69.353271  6.769671e-03  8.418663e-03
##        B -410.632837  9.829578e-06  1.347119e-05
##        S   -5.651611  5.440489e-04  6.820738e-04
##      A+B  -20.037251 -1.033065e-03 -4.682618e-03
##      A+S   -1.919434  7.873469e-05  9.025960e-05
##      B+S   14.103789 -2.865330e-04 -5.188439e-04
##    A+B+S   -7.352686  3.539970e-04 -9.123065e-05
##     Full -500.843301  6.436684e-03  3.911775e-03
```

```r
plot(an)
```

![](README_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

The anova shows the relative changes in the logLik of the groups and their intersections:

Space has a high positive value which means that space does not increase the model fit.



If it fails, check out the help of ?install_sjSDM, ?installation_help, and vignette("Dependencies", package = "sjSDM"). 

#### Installation workflow:
1. Try install_sjSDM()
2. New session, if no 'PyTorch not found' appears it should work, otherwise see ?installation_help
3. If do not get the pkg to run, create an issue [issue tracker](https://github.com/TheoreticalEcology/s-jSDM/issues) or write an email to maximilian.pichler at ur.de


### Python Package

```bash
pip install sjSDM_py
```
Python example


```python
import sjSDM_py as fa
import numpy as np
Env = np.random.randn(100, 5)
Occ = np.random.binomial(1, 0.5, [100, 10])

model = fa.Model_base(5)
model.add_layer(fa.layers.Layer_dense(10))
model.build(df=5, optimizer=fa.optimizer_adamax(0.1))
model.fit(X = Env, Y = Occ)
print(model.weights_numpy)
print(model.get_cov())
```

