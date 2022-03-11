
<!-- README.md is generated from README.Rmd. Please edit that file -->

[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![License: GPL
v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/sjSDM)](https://cran.r-project.org/package=sjSDM)
![R-CMD-check](https://github.com/TheoreticalEcology/s-jSDM/workflows/R-CMD-check/badge.svg?branch=master)
[![Publication](https://img.shields.io/badge/Publication-10.1111/2041-green.svg)](https://besjournals.onlinelibrary.wiley.com/doi/abs/10.1111/2041-210X.13687)

# s-jSDM - Fast and accurate Joint Species Distribution Modeling

## About the method

The method is described in Pichler & Hartig (2021) A new joint species
distribution model for faster and more accurate inference of species
associations from big community data,
<https://doi.org/10.1111/2041-210X.13687>. The code for producing the
results in this paper is available under the subfolder publications in
this repo.

The method itself is wrapped into an R package, available under
subfolder sjSDM. You can also use it stand-alone under Python (see
instructions below). Note: for both the R and the python package, python
\>= 3.6 and pytorch must be installed (more details below).

## Installing the R / Python package

### R-package

Install the package via

``` r
install.packages("sjSDM")
```

Depencies for the package can be installed before or after installing
the package. Detailed explanations of the dependencies are provided in
vignette(“Dependencies”, package = “sjSDM”), source code
[here](https://github.com/TheoreticalEcology/s-jSDM/blob/master/sjSDM/vignettes/Dependencies.Rmd).
Very briefly, the dependencies can be automatically installed from
within R:

``` r
sjSDM::install_sjSDM(version = "gpu") # or
sjSDM::install_sjSDM(version = "cpu")
```

To cite sjSDM, please use the following citation:

``` r
citation("sjSDM")
```

### Development

If you want to install the current (development) version from this
repository, run

``` r
devtools::install_github("https://github.com/TheoreticalEcology/s-jSDM", subdir = "sjSDM", ref = "devel")
```

Once the dependencies are installed, the following code should run:

Simulate a community and fit model:

``` r
library(sjSDM)
```

    ## ── Attaching sjSDM ──────────────────────────────────────────────────── 1.0.1 ──

    ## ✓ torch <environment> 
    ## ✓ torch_optimizer  
    ## ✓ pyro  
    ## ✓ madgrad

``` r
set.seed(42)
community <- simulate_SDM(sites = 100, species = 10, env = 3, se = TRUE)
Env <- community$env_weights
Occ <- community$response
SP <- matrix(rnorm(200, 0, 0.3), 100, 2) # spatial coordinates (no effect on species occurences)

model <- sjSDM(Y = Occ, env = linear(data = Env, formula = ~X1+X2+X3), spatial = linear(data = SP, formula = ~0+X1:X2), se = TRUE, family=binomial("probit"), sampling = 100L)
summary(model)
```

    ## LogLik:  -517.1162 
    ## Regularization loss:  0 
    ## 
    ## Species-species correlation matrix: 
    ## 
    ##  sp1  1.0000                                 
    ##  sp2 -0.3460  1.0000                             
    ##  sp3 -0.1400 -0.3770  1.0000                         
    ##  sp4 -0.1410 -0.3500  0.7070  1.0000                     
    ##  sp5  0.6090 -0.3260 -0.0870 -0.0940  1.0000                 
    ##  sp6 -0.2180  0.3640  0.1600  0.1490 -0.0550  1.0000             
    ##  sp7  0.4700 -0.1420  0.1320  0.1190  0.5120  0.2390  1.0000         
    ##  sp8  0.2530  0.1540 -0.4360 -0.4220  0.2340 -0.0540  0.1100  1.0000     
    ##  sp9 -0.1060 -0.0420  0.0510  0.0610 -0.2900 -0.2650 -0.2250 -0.1180  1.0000 
    ##  sp10     0.2070  0.3830 -0.6020 -0.5750  0.2350  0.0820  0.1270  0.4150 -0.2100  1.0000
    ## 
    ## 
    ## 
    ## Spatial: 
    ##             sp1        sp2       sp3         sp4       sp5       sp6       sp7
    ## X1:X2 0.1831122 -0.5116199 0.5413426 -0.00264241 0.3427651 0.1870106 0.4059039
    ##             sp8       sp9      sp10
    ## X1:X2 0.4804094 0.2659622 0.2485598
    ## 
    ## 
    ## 
    ##                  Estimate  Std.Err Z value Pr(>|z|)    
    ## sp1 (Intercept)  -0.02294  0.20454   -0.11  0.91068    
    ## sp1 X1            0.83191  0.40909    2.03  0.04199 *  
    ## sp1 X2           -1.63252  0.37725   -4.33  1.5e-05 ***
    ## sp1 X3           -0.12155  0.33693   -0.36  0.71828    
    ## sp2 (Intercept)  -0.03995  0.22700   -0.18  0.86029    
    ## sp2 X1            0.89410  0.41314    2.16  0.03045 *  
    ## sp2 X2            0.15840  0.40862    0.39  0.69827    
    ## sp2 X3            0.43637  0.39043    1.12  0.26371    
    ## sp3 (Intercept)  -0.34723  0.22196   -1.56  0.11773    
    ## sp3 X1            1.03418  0.40596    2.55  0.01085 *  
    ## sp3 X2           -0.21915  0.40775   -0.54  0.59095    
    ## sp3 X3           -0.68271  0.38546   -1.77  0.07654 .  
    ## sp4 (Intercept)  -0.04001  0.20259   -0.20  0.84345    
    ## sp4 X1           -1.02553  0.38775   -2.64  0.00817 ** 
    ## sp4 X2           -1.27792  0.38960   -3.28  0.00104 ** 
    ## sp4 X3           -0.27332  0.34129   -0.80  0.42323    
    ## sp5 (Intercept)  -0.13282  0.19686   -0.67  0.49986    
    ## sp5 X1            0.45864  0.37292    1.23  0.21875    
    ## sp5 X2            0.42704  0.36548    1.17  0.24264    
    ## sp5 X3           -0.40132  0.34380   -1.17  0.24308    
    ## sp6 (Intercept)   0.17045  0.19915    0.86  0.39206    
    ## sp6 X1            1.68681  0.40498    4.17  3.1e-05 ***
    ## sp6 X2           -0.66825  0.38106   -1.75  0.07949 .  
    ## sp6 X3            0.17839  0.33066    0.54  0.58954    
    ## sp7 (Intercept)   0.00701  0.20640    0.03  0.97289    
    ## sp7 X1           -0.28037  0.40207   -0.70  0.48560    
    ## sp7 X2            0.29147  0.37519    0.78  0.43725    
    ## sp7 X3           -1.06023  0.35214   -3.01  0.00261 ** 
    ## sp8 (Intercept)   0.13752  0.14967    0.92  0.35816    
    ## sp8 X1            0.26011  0.28608    0.91  0.36323    
    ## sp8 X2            0.27259  0.27891    0.98  0.32840    
    ## sp8 X3           -1.01502  0.26326   -3.86  0.00012 ***
    ## sp9 (Intercept)   0.04048  0.16892    0.24  0.81061    
    ## sp9 X1            1.10252  0.32760    3.37  0.00076 ***
    ## sp9 X2           -0.81973  0.31488   -2.60  0.00923 ** 
    ## sp9 X3            0.59836  0.28592    2.09  0.03637 *  
    ## sp10 (Intercept) -0.09065  0.18045   -0.50  0.61542    
    ## sp10 X1          -0.43825  0.33418   -1.31  0.18972    
    ## sp10 X2          -0.94811  0.32707   -2.90  0.00375 ** 
    ## sp10 X3          -0.37165  0.30311   -1.23  0.22015    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
plot(model)
```

    ## LogLik:  -517.1162 
    ## Regularization loss:  0 
    ## 
    ## Species-species correlation matrix: 
    ## 
    ##  sp1  1.0000                                 
    ##  sp2 -0.3460  1.0000                             
    ##  sp3 -0.1400 -0.3770  1.0000                         
    ##  sp4 -0.1410 -0.3500  0.7070  1.0000                     
    ##  sp5  0.6090 -0.3260 -0.0870 -0.0940  1.0000                 
    ##  sp6 -0.2180  0.3640  0.1600  0.1490 -0.0550  1.0000             
    ##  sp7  0.4700 -0.1420  0.1320  0.1190  0.5120  0.2390  1.0000         
    ##  sp8  0.2530  0.1540 -0.4360 -0.4220  0.2340 -0.0540  0.1100  1.0000     
    ##  sp9 -0.1060 -0.0420  0.0510  0.0610 -0.2900 -0.2650 -0.2250 -0.1180  1.0000 
    ##  sp10     0.2070  0.3830 -0.6020 -0.5750  0.2350  0.0820  0.1270  0.4150 -0.2100  1.0000
    ## 
    ## 
    ## 
    ## Spatial: 
    ##             sp1        sp2       sp3         sp4       sp5       sp6       sp7
    ## X1:X2 0.1831122 -0.5116199 0.5413426 -0.00264241 0.3427651 0.1870106 0.4059039
    ##             sp8       sp9      sp10
    ## X1:X2 0.4804094 0.2659622 0.2485598
    ## 
    ## 
    ## 
    ##                  Estimate  Std.Err Z value Pr(>|z|)    
    ## sp1 (Intercept)  -0.02294  0.20454   -0.11  0.91068    
    ## sp1 X1            0.83191  0.40909    2.03  0.04199 *  
    ## sp1 X2           -1.63252  0.37725   -4.33  1.5e-05 ***
    ## sp1 X3           -0.12155  0.33693   -0.36  0.71828    
    ## sp2 (Intercept)  -0.03995  0.22700   -0.18  0.86029    
    ## sp2 X1            0.89410  0.41314    2.16  0.03045 *  
    ## sp2 X2            0.15840  0.40862    0.39  0.69827    
    ## sp2 X3            0.43637  0.39043    1.12  0.26371    
    ## sp3 (Intercept)  -0.34723  0.22196   -1.56  0.11773    
    ## sp3 X1            1.03418  0.40596    2.55  0.01085 *  
    ## sp3 X2           -0.21915  0.40775   -0.54  0.59095    
    ## sp3 X3           -0.68271  0.38546   -1.77  0.07654 .  
    ## sp4 (Intercept)  -0.04001  0.20259   -0.20  0.84345    
    ## sp4 X1           -1.02553  0.38775   -2.64  0.00817 ** 
    ## sp4 X2           -1.27792  0.38960   -3.28  0.00104 ** 
    ## sp4 X3           -0.27332  0.34129   -0.80  0.42323    
    ## sp5 (Intercept)  -0.13282  0.19686   -0.67  0.49986    
    ## sp5 X1            0.45864  0.37292    1.23  0.21875    
    ## sp5 X2            0.42704  0.36548    1.17  0.24264    
    ## sp5 X3           -0.40132  0.34380   -1.17  0.24308    
    ## sp6 (Intercept)   0.17045  0.19915    0.86  0.39206    
    ## sp6 X1            1.68681  0.40498    4.17  3.1e-05 ***
    ## sp6 X2           -0.66825  0.38106   -1.75  0.07949 .  
    ## sp6 X3            0.17839  0.33066    0.54  0.58954    
    ## sp7 (Intercept)   0.00701  0.20640    0.03  0.97289    
    ## sp7 X1           -0.28037  0.40207   -0.70  0.48560    
    ## sp7 X2            0.29147  0.37519    0.78  0.43725    
    ## sp7 X3           -1.06023  0.35214   -3.01  0.00261 ** 
    ## sp8 (Intercept)   0.13752  0.14967    0.92  0.35816    
    ## sp8 X1            0.26011  0.28608    0.91  0.36323    
    ## sp8 X2            0.27259  0.27891    0.98  0.32840    
    ## sp8 X3           -1.01502  0.26326   -3.86  0.00012 ***
    ## sp9 (Intercept)   0.04048  0.16892    0.24  0.81061    
    ## sp9 X1            1.10252  0.32760    3.37  0.00076 ***
    ## sp9 X2           -0.81973  0.31488   -2.60  0.00923 ** 
    ## sp9 X3            0.59836  0.28592    2.09  0.03637 *  
    ## sp10 (Intercept) -0.09065  0.18045   -0.50  0.61542    
    ## sp10 X1          -0.43825  0.33418   -1.31  0.18972    
    ## sp10 X2          -0.94811  0.32707   -2.90  0.00375 ** 
    ## sp10 X3          -0.37165  0.30311   -1.23  0.22015    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

We also support also other response families: Count data:

``` r
model <- sjSDM(Y = Occ, env = linear(data = Env, formula = ~X1+X2+X3), spatial = linear(data = SP, formula = ~0+X1:X2), se = TRUE, family=poisson("log"))
```

Gaussian (normal):

``` r
model <- sjSDM(Y = Occ, env = linear(data = Env, formula = ~X1+X2+X3), spatial = linear(data = SP, formula = ~0+X1:X2), se = TRUE, family=gaussian("identity"))
```

Let’s have a look at the importance of the three groups (environment,
associations, and space) on the occurences:

``` r
imp = importance(model)
```

    ## Warning: 'importance' is deprecated.
    ## Use 'plot(anova(x, internal=TRUE))' instead.
    ## See help("Deprecated")

``` r
print(imp)
```

    ##    sp       env      spatial     biotic
    ## 1   1 0.8661759 2.439290e-04 0.13358012
    ## 2   2 0.5512735 3.389213e-03 0.44533733
    ## 3   3 0.7243540 3.892151e-03 0.27175385
    ## 4   4 0.8974715 5.453921e-08 0.10252855
    ## 5   5 0.4491234 2.478703e-03 0.54839790
    ## 6   6 0.8595459 2.568011e-04 0.14019726
    ## 7   7 0.5315694 1.515744e-03 0.46691483
    ## 8   8 0.9476700 4.443077e-03 0.04788696
    ## 9   9 0.7424834 6.183612e-04 0.25689825
    ## 10 10 0.9604391 1.141932e-03 0.03841905

``` r
plot(imp)
```

![](README_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

As expected, space has no effect on occurences.

Let’s have a look on community level how the three groups contribute to
the overall explained variance

``` r
an = anova(model)
print(an)
```

    ## Analysis of Deviance Table
    ## 
    ## Terms added sequentially:
    ## 
    ##          Deviance Residual deviance R2 Nagelkerke R2 McFadden
    ## Biotic  153.69963         518.82466       0.78497      0.1112
    ## Abiotic 180.74306         338.08160       0.96472      0.2419
    ## Spatial  12.84565         325.23595       0.96897      0.2512

``` r
plot(an)
```

![](README_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

The anova shows the relative changes in the deviance of the groups and
their intersections.

We can also visualize the individual contributions to the species and
site
*R*<sup>2</sup>
:

``` r
plot(an, internal=TRUE)
```

    ## Registered S3 methods overwritten by 'ggtern':
    ##   method           from   
    ##   grid.draw.ggplot ggplot2
    ##   plot.ggplot      ggplot2
    ##   print.ggplot     ggplot2

![](README_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

If it fails, check out the help of ?install_sjSDM, ?installation_help,
and vignette(“Dependencies”, package = “sjSDM”).

#### Installation workflow:

1.  Try install_sjSDM()
2.  New session, if no ‘PyTorch not found’ appears it should work,
    otherwise see ?installation_help
3.  If do not get the pkg to run, create an issue [issue
    tracker](https://github.com/TheoreticalEcology/s-jSDM/issues) or
    write an email to maximilian.pichler at ur.de

### Python Package

``` bash
pip install sjSDM_py
```

Python example

``` python
import sjSDM_py as fa
import numpy as np
import torch
Env = np.random.randn(100, 5)
Occ = np.random.binomial(1, 0.5, [100, 10])

model = fa.Model_sjSDM(device=torch.device("cpu"), dtype=torch.float32)
model.add_env(5, 10)
model.build(5, optimizer=fa.optimizer_adamax(0.001),scheduler=False)
model.fit(Env, Occ, batch_size = 20, epochs = 10)
# print(model.weights)
# print(model.covariance)
```

    ## Iter: 0/10   0%|          | [00:00, ?it/s]Iter: 0/10   0%|          | [00:00, ?it/s, loss=7.233]Iter: 1/10  10%|#         | [00:00,  5.80it/s, loss=7.233]Iter: 1/10  10%|#         | [00:00,  5.80it/s, loss=7.237]Iter: 1/10  10%|#         | [00:00,  5.80it/s, loss=7.222]Iter: 1/10  10%|#         | [00:00,  5.80it/s, loss=7.204]Iter: 1/10  10%|#         | [00:00,  5.80it/s, loss=7.22] Iter: 1/10  10%|#         | [00:00,  5.80it/s, loss=7.2] Iter: 1/10  10%|#         | [00:00,  5.80it/s, loss=7.188]Iter: 7/10  70%|#######   | [00:00, 30.03it/s, loss=7.188]Iter: 7/10  70%|#######   | [00:00, 30.03it/s, loss=7.196]Iter: 7/10  70%|#######   | [00:00, 30.03it/s, loss=7.187]Iter: 7/10  70%|#######   | [00:00, 30.03it/s, loss=7.192]Iter: 10/10 100%|##########| [00:00, 30.66it/s, loss=7.192]

Calculate Importance:

``` python
Beta = np.transpose(model.env_weights[0])
Sigma = ( model.sigma @ model.sigma.t() + torch.diag(torch.ones([1])) ).data.cpu().numpy()
covX = fa.covariance( torch.tensor(Env).t() ).data.cpu().numpy()

fa.importance(beta=Beta, covX=covX, sigma=Sigma)
```

    ## {'env': array([[ 8.01924523e-03,  5.88351442e-03,  3.46450630e-04,
    ##          2.41728313e-03,  2.93385610e-03],
    ##        [ 2.51798570e-04,  1.30314089e-03,  4.08564508e-03,
    ##          8.43194837e-04,  1.00540789e-02],
    ##        [ 4.58702561e-04,  6.93971897e-03,  9.88384054e-05,
    ##          1.51673518e-03,  4.04162938e-03],
    ##        [-1.49338421e-05,  3.24389548e-05,  1.47342135e-03,
    ##          3.60254053e-05,  7.97839928e-03],
    ##        [ 7.79314898e-04,  1.94915745e-04,  6.61612488e-03,
    ##          1.23276515e-02,  3.61960870e-03],
    ##        [ 3.79100814e-03,  9.13846270e-06,  7.05215894e-03,
    ##          8.15463625e-03,  2.77351518e-03],
    ##        [ 3.37206497e-04,  9.80060431e-04,  6.51973824e-05,
    ##          1.43132071e-04,  1.15610389e-02],
    ##        [ 6.48319907e-03,  6.12059329e-03,  9.50422324e-03,
    ##          8.78254510e-03,  9.26101953e-03],
    ##        [ 8.99332296e-03,  1.01236743e-03,  3.25980218e-04,
    ##          1.23339929e-02,  1.86911039e-03],
    ##        [ 8.95618182e-03,  1.53849716e-03,  1.09001040e-03,
    ##          6.02281233e-03,  7.11724581e-03]], dtype=float32), 'biotic': array([0.98039967, 0.98346215, 0.9869444 , 0.9904946 , 0.9764624 ,
    ##        0.97821957, 0.9869134 , 0.9598484 , 0.97546524, 0.9752752 ],
    ##       dtype=float32)}
