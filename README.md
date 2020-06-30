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
set.seed(42)
community <- simulate_SDM(sites = 400, species = 10, env = 3)
Env <- community$env_weights
Occ <- community$response
SP <- matrix(rnorm(800, 0, 0.3), 400, 2) # spatial coordinates (no effect on species occurences)

model <- sjSDM(Y = Occ, env = linear(data = Env, formula = ~X1+X2+X3), spatial = linear(data = SP, formula = ~0+X1:X2), se = TRUE)
summary(model)
```

```
## LogLik:  -2067.809 
## Deviance:  4135.617 
## 
## Regularization loss:  0 
## 
## Species-species correlation matrix: 
## 
##         sp1    sp2    sp3    sp4    sp5    sp6    sp7    sp8    sp9 sp10
## sp1   1.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000    0
## sp2   0.804  1.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000    0
## sp3   0.730  0.769  1.000  0.000  0.000  0.000  0.000  0.000  0.000    0
## sp4  -0.351 -0.500 -0.729  1.000  0.000  0.000  0.000  0.000  0.000    0
## sp5   0.622  0.755  0.521 -0.718  1.000  0.000  0.000  0.000  0.000    0
## sp6   0.482  0.838  0.605 -0.613  0.792  1.000  0.000  0.000  0.000    0
## sp7  -0.031  0.002  0.311 -0.743  0.412  0.383  1.000  0.000  0.000    0
## sp8   0.370 -0.014 -0.084  0.672 -0.405 -0.456 -0.754  1.000  0.000    0
## sp9  -0.322  0.095  0.117 -0.634  0.381  0.310  0.398 -0.845  1.000    0
## sp10  0.351  0.421  0.682 -0.135 -0.156  0.295 -0.032  0.227 -0.278    1
## 
## 
## 
## Spatial: 
##             sp1        sp2       sp3        sp4       sp5         sp6
## X1:X2 0.1148056 -0.1612237 0.2041012 -0.2939318 0.2416552 -0.02739704
##               sp7        sp8          sp9       sp10
## X1:X2 -0.02912659 -0.1707652 -0.006200577 -0.2947339
## 
## 
## 
##                  Estimate Std.Err Z value Pr(>|z|)    
## sp1 (Intercept)    0.0117  0.1027    0.11  0.90897    
## sp1 X1             1.5283  0.1883    8.11  4.9e-16 ***
## sp1 X2             1.0569  0.1793    5.89  3.8e-09 ***
## sp1 X3             0.1394  0.1699    0.82  0.41178    
## sp2 (Intercept)   -0.0748  0.1025   -0.73  0.46559    
## sp2 X1             1.6368  0.1890    8.66  < 2e-16 ***
## sp2 X2             0.3360  0.1803    1.86  0.06238 .  
## sp2 X3            -0.6501  0.1726   -3.77  0.00017 ***
## sp3 (Intercept)    0.1281  0.1030    1.24  0.21352    
## sp3 X1            -0.6049  0.1786   -3.39  0.00071 ***
## sp3 X2            -0.6181  0.1803   -3.43  0.00061 ***
## sp3 X3             0.7452  0.1752    4.25  2.1e-05 ***
## sp4 (Intercept)    0.1375  0.0944    1.46  0.14517    
## sp4 X1            -1.2030  0.1698   -7.09  1.4e-12 ***
## sp4 X2             0.7050  0.1713    4.12  3.9e-05 ***
## sp4 X3             0.4287  0.1567    2.74  0.00622 ** 
## sp5 (Intercept)   -0.0908  0.0984   -0.92  0.35593    
## sp5 X1            -1.0677  0.1729   -6.18  6.6e-10 ***
## sp5 X2            -0.8738  0.1728   -5.06  4.3e-07 ***
## sp5 X3            -0.1183  0.1676   -0.71  0.48018    
## sp6 (Intercept)   -0.0999  0.0896   -1.11  0.26519    
## sp6 X1             0.6395  0.1564    4.09  4.3e-05 ***
## sp6 X2             1.2251  0.1616    7.58  3.5e-14 ***
## sp6 X3             0.6403  0.1561    4.10  4.1e-05 ***
## sp7 (Intercept)    0.0472  0.0795    0.59  0.55264    
## sp7 X1            -0.0171  0.1368   -0.12  0.90081    
## sp7 X2             0.7501  0.1392    5.39  7.2e-08 ***
## sp7 X3            -1.0855  0.1402   -7.74  9.6e-15 ***
## sp8 (Intercept)    0.2478  0.0962    2.57  0.01004 *  
## sp8 X1            -0.6846  0.1671   -4.10  4.2e-05 ***
## sp8 X2             0.9094  0.1731    5.25  1.5e-07 ***
## sp8 X3             0.4883  0.1594    3.06  0.00219 ** 
## sp9 (Intercept)    0.0558  0.0801    0.70  0.48580    
## sp9 X1            -0.6817  0.1424   -4.79  1.7e-06 ***
## sp9 X2            -0.9264  0.1447   -6.40  1.5e-10 ***
## sp9 X3            -0.8629  0.1390   -6.21  5.3e-10 ***
## sp10 (Intercept)   0.0958  0.0874    1.10  0.27273    
## sp10 X1           -0.5955  0.1533   -3.88  0.00010 ***
## sp10 X2           -0.3735  0.1516   -2.46  0.01376 *  
## sp10 X3           -0.8907  0.1493   -5.96  2.5e-09 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```
Let's have a look at the importance of the three groups (environment, associations, and space) on the occurences:

```r
imp = importance(model)
print(imp)
```

```
##    sp        env      spatial    biotic
## 1   1 0.20932963 2.109202e-05 0.7906493
## 2   2 0.17131556 3.515757e-05 0.8286493
## 3   3 0.06092341 4.979560e-05 0.9390268
## 4   4 0.16444470 1.705628e-04 0.8353847
## 5   5 0.11954581 9.659126e-05 0.8803576
## 6   6 0.21237895 1.786013e-06 0.7876193
## 7   7 0.34098728 4.458012e-06 0.6590083
## 8   8 0.15573741 7.463750e-05 0.8441880
## 9   9 0.41143604 1.979391e-07 0.5885638
## 10 10 0.19087143 3.374785e-04 0.8087911
```

```r
plot(imp)
```

![](README_files/figure-html/unnamed-chunk-4-1.png)<!-- -->


As expected, space has no effect on occurences.

Let's have a look on community level how the three groups contribute to the overall explained variance 

```r
an = anova(model)
print(an)
```

```
## Changes relative to empty model (without modules):
## 
##  Modules       LogLik            R2   marginal R2          R2ll
##        _ 2772.5878906  0.000000e+00  0.000000e+00  0.0000000000
##        A -380.0660095  3.415679e-02  2.183816e-02  0.1370798779
##        B -245.7903961  7.632231e-06  5.352274e-06  0.0886501730
##        S   -0.3488159  1.138984e-04  7.103606e-05  0.0001258088
##      A+B  -16.6191071 -3.002190e-03 -9.996732e-03  0.0059940776
##      A+S    0.6522522 -9.303914e-05 -2.961246e-05 -0.0002352503
##      B+S   -0.5339062 -7.842866e-05 -5.864212e-05  0.0001925660
##    A+B+S    0.7018323 -3.098628e-05  2.771568e-05 -0.0002531326
##     Full -642.0041504  3.107368e-02  1.185728e-02  0.2315541205
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
import torch
Env = np.random.randn(100, 5)
Occ = np.random.binomial(1, 0.5, [100, 10])

model = fa.Model_sjSDM(device=torch.device("cpu"), dtype=torch.float32)
model.add_env(5, 10)
model.build(5, optimizer=fa.optimizer_adamax(0.001),scheduler=False)
model.fit(Env, Occ, batch_size = 20, epochs = 100)
# print(model.weights)
# print(model.covariance)
```

```
## 
Iter: 0/100   0%|          | [00:00, ?it/s]
Iter: 100/100 100%|##########| [00:02, 37.95it/s, loss=7.096]
```

