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
## LogLik:  -2039.769 
## Deviance:  4079.538 
## 
## Regularization loss:  0 
## 
## Species-species correlation matrix: 
## 
##         sp1    sp2    sp3    sp4    sp5    sp6    sp7    sp8    sp9 sp10
## sp1   1.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000    0
## sp2   0.729  1.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000    0
## sp3   0.697  0.727  1.000  0.000  0.000  0.000  0.000  0.000  0.000    0
## sp4  -0.329 -0.436 -0.732  1.000  0.000  0.000  0.000  0.000  0.000    0
## sp5   0.609  0.710  0.482 -0.675  1.000  0.000  0.000  0.000  0.000    0
## sp6   0.444  0.824  0.565 -0.567  0.780  1.000  0.000  0.000  0.000    0
## sp7   0.017 -0.119  0.279 -0.705  0.368  0.294  1.000  0.000  0.000    0
## sp8   0.421  0.023 -0.041  0.625 -0.363 -0.442 -0.669  1.000  0.000    0
## sp9  -0.336  0.151  0.155 -0.602  0.335  0.283  0.216 -0.765  1.000    0
## sp10  0.312  0.446  0.682 -0.149 -0.168  0.306 -0.070  0.208 -0.193    1
## 
## 
## 
## Spatial: 
##             sp1        sp2      sp3        sp4       sp5          sp6
## X1:X2 0.1552027 -0.1965767 0.252658 -0.3712009 0.3397274 -0.003926652
##               sp7        sp8        sp9       sp10
## X1:X2 -0.01596577 -0.2407501 0.03388538 -0.3567291
## 
## 
## 
##                  Estimate Std.Err Z value Pr(>|z|)    
## sp1 (Intercept)    0.0213  0.0691    0.31  0.75847    
## sp1 X1             1.1605  0.1268    9.15  < 2e-16 ***
## sp1 X2             0.8179  0.1224    6.68  2.4e-11 ***
## sp1 X3             0.1020  0.1142    0.89  0.37185    
## sp2 (Intercept)   -0.0269  0.0705   -0.38  0.70223    
## sp2 X1             1.2598  0.1329    9.48  < 2e-16 ***
## sp2 X2             0.2615  0.1221    2.14  0.03213 *  
## sp2 X3            -0.4953  0.1193   -4.15  3.3e-05 ***
## sp3 (Intercept)    0.1152  0.0692    1.66  0.09605 .  
## sp3 X1            -0.4251  0.1197   -3.55  0.00038 ***
## sp3 X2            -0.4535  0.1201   -3.78  0.00016 ***
## sp3 X3             0.5355  0.1184    4.52  6.1e-06 ***
## sp4 (Intercept)    0.0857  0.0587    1.46  0.14414    
## sp4 X1            -0.8385  0.1066   -7.86  3.7e-15 ***
## sp4 X2             0.4959  0.1081    4.59  4.5e-06 ***
## sp4 X3             0.3098  0.0970    3.19  0.00141 ** 
## sp5 (Intercept)   -0.0327  0.0659   -0.50  0.61991    
## sp5 X1            -0.7478  0.1165   -6.42  1.4e-10 ***
## sp5 X2            -0.6217  0.1172   -5.31  1.1e-07 ***
## sp5 X3            -0.1126  0.1138   -0.99  0.32259    
## sp6 (Intercept)   -0.0493  0.0573   -0.86  0.38945    
## sp6 X1             0.4589  0.0996    4.61  4.1e-06 ***
## sp6 X2             0.8463  0.1032    8.20  2.4e-16 ***
## sp6 X3             0.4314  0.1000    4.31  1.6e-05 ***
## sp7 (Intercept)    0.0339  0.0509    0.67  0.50558    
## sp7 X1            -0.0175  0.0875   -0.20  0.84149    
## sp7 X2             0.4953  0.0901    5.50  3.8e-08 ***
## sp7 X3            -0.7395  0.0899   -8.23  < 2e-16 ***
## sp8 (Intercept)    0.1566  0.0609    2.57  0.01015 *  
## sp8 X1            -0.4679  0.1077   -4.34  1.4e-05 ***
## sp8 X2             0.6396  0.1108    5.77  7.9e-09 ***
## sp8 X3             0.3304  0.1009    3.27  0.00106 ** 
## sp9 (Intercept)    0.0372  0.0499    0.75  0.45560    
## sp9 X1            -0.4497  0.0892   -5.04  4.7e-07 ***
## sp9 X2            -0.6199  0.0904   -6.86  7.0e-12 ***
## sp9 X3            -0.5670  0.0871   -6.51  7.5e-11 ***
## sp10 (Intercept)   0.0629  0.0557    1.13  0.25921    
## sp10 X1           -0.3893  0.0987   -3.94  8.1e-05 ***
## sp10 X2           -0.2474  0.0970   -2.55  0.01072 *  
## sp10 X3           -0.5924  0.0952   -6.22  4.9e-10 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```
Let's have a look at the importance of the three groups (environment, associations, and space) on the occurences:

```r
imp = importance(model)
print(imp)
```

```
##    sp       env      spatial    biotic
## 1   1 0.4224785 1.333015e-04 0.5773882
## 2   2 0.3456836 1.784461e-04 0.6541380
## 3   3 0.1341345 3.258612e-04 0.8655397
## 4   4 0.3857035 1.297156e-03 0.6129993
## 5   5 0.2616182 8.363503e-04 0.7375455
## 6   6 0.4219868 1.521797e-07 0.5780130
## 7   7 0.6940636 5.985971e-06 0.3059304
## 8   8 0.4037986 8.020784e-04 0.5953993
## 9   9 0.7780283 2.545462e-05 0.2219462
## 10 10 0.4180111 2.470536e-03 0.5795183
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
##        _ 2772.5878906  0.000000e+00  0.000000e+00  0.000000e+00
##        A -378.6965942  3.621014e-02  4.528958e-02  1.365860e-01
##        B -249.2356421  8.010719e-06  1.178901e-05  8.989278e-02
##        S    0.8609619  3.353837e-04  4.185826e-04 -3.105265e-04
##      A+B  -27.3135065 -2.376274e-03 -2.079829e-02  9.851268e-03
##      A+S    0.1219482 -1.303424e-04 -2.208498e-04 -4.398354e-05
##      B+S    4.1789069 -2.134936e-04 -3.295743e-04 -1.507223e-03
##    A+B+S   -3.1991650 -6.752495e-05  1.245954e-04  1.153855e-03
##     Full -653.2830908  3.376590e-02  2.449584e-02  2.356221e-01
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

