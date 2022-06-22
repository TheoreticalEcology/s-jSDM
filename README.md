
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

    ## ── Attaching sjSDM ──────────────────────────────────────────────────── 1.0.2 ──

    ## ✔ torch <environment> 
    ## ✔ torch_optimizer  
    ## ✔ pyro  
    ## ✔ madgrad

``` r
set.seed(42)
community <- simulate_SDM(sites = 100, species = 10, env = 3, se = TRUE)
Env <- community$env_weights
Occ <- community$response
SP <- matrix(rnorm(200, 0, 0.3), 100, 2) # spatial coordinates (no effect on species occurences)

model <- sjSDM(Y = Occ, env = linear(data = Env, formula = ~X1+X2+X3), spatial = linear(data = SP, formula = ~0+X1:X2), se = TRUE, family=binomial("probit"), sampling = 100L)
summary(model)
```

    ## LogLik:  -511.7972 
    ## Regularization loss:  0 
    ## 
    ## Species-species correlation matrix: 
    ## 
    ##  sp1  1.0000                                 
    ##  sp2 -0.3560  1.0000                             
    ##  sp3 -0.1180 -0.3740  1.0000                         
    ##  sp4 -0.1220 -0.3350  0.6950  1.0000                     
    ##  sp5  0.5970 -0.3510 -0.0760 -0.0870  1.0000                 
    ##  sp6 -0.2140  0.3880  0.1420  0.1350 -0.0730  1.0000             
    ##  sp7  0.4640 -0.1540  0.1470  0.1390  0.5120  0.2440  1.0000         
    ##  sp8  0.2560  0.1570 -0.4240 -0.4180  0.2150 -0.0320  0.1070  1.0000     
    ##  sp9 -0.0730 -0.0470  0.0540  0.0700 -0.3050 -0.2550 -0.2220 -0.1010  1.0000 
    ##  sp10     0.1800  0.3920 -0.6130 -0.5760  0.2220  0.1000  0.1090  0.4120 -0.2190  1.0000
    ## 
    ## 
    ## 
    ## Spatial: 
    ##             sp1        sp2       sp3        sp4       sp5       sp6       sp7
    ## X1:X2 0.2260564 -0.5308802 0.5423007 0.03226206 0.3345731 0.1966482 0.4499481
    ##             sp8       sp9      sp10
    ## X1:X2 0.4780776 0.2826627 0.2449929
    ## 
    ## 
    ## 
    ##                  Estimate Std.Err Z value Pr(>|z|)    
    ## sp1 (Intercept)   -0.0290  0.2081   -0.14  0.88925    
    ## sp1 X1             0.8329  0.3936    2.12  0.03433 *  
    ## sp1 X2            -1.6153  0.3787   -4.27  2.0e-05 ***
    ## sp1 X3            -0.1006  0.3489   -0.29  0.77308    
    ## sp2 (Intercept)   -0.0335  0.2164   -0.16  0.87681    
    ## sp2 X1             0.8809  0.4079    2.16  0.03080 *  
    ## sp2 X2             0.1921  0.3994    0.48  0.63055    
    ## sp2 X3             0.4257  0.3576    1.19  0.23383    
    ## sp3 (Intercept)   -0.3370  0.2195   -1.54  0.12472    
    ## sp3 X1             1.0309  0.4044    2.55  0.01080 *  
    ## sp3 X2            -0.2759  0.3924   -0.70  0.48190    
    ## sp3 X3            -0.7200  0.3764   -1.91  0.05577 .  
    ## sp4 (Intercept)   -0.0535  0.1962   -0.27  0.78510    
    ## sp4 X1            -1.0306  0.3724   -2.77  0.00565 ** 
    ## sp4 X2            -1.2974  0.3783   -3.43  0.00061 ***
    ## sp4 X3            -0.3160  0.3288   -0.96  0.33662    
    ## sp5 (Intercept)   -0.1503  0.2044   -0.74  0.46211    
    ## sp5 X1             0.5085  0.3914    1.30  0.19383    
    ## sp5 X2             0.4517  0.3673    1.23  0.21880    
    ## sp5 X3            -0.4063  0.3406   -1.19  0.23291    
    ## sp6 (Intercept)    0.1879  0.2058    0.91  0.36116    
    ## sp6 X1             1.6793  0.4291    3.91  9.1e-05 ***
    ## sp6 X2            -0.6698  0.3940   -1.70  0.08912 .  
    ## sp6 X3             0.1595  0.3369    0.47  0.63595    
    ## sp7 (Intercept)   -0.0188  0.2002   -0.09  0.92511    
    ## sp7 X1            -0.2593  0.3889   -0.67  0.50501    
    ## sp7 X2             0.3116  0.3679    0.85  0.39698    
    ## sp7 X3            -1.0443  0.3462   -3.02  0.00255 ** 
    ## sp8 (Intercept)    0.1435  0.1533    0.94  0.34911    
    ## sp8 X1             0.2595  0.2950    0.88  0.37895    
    ## sp8 X2             0.3017  0.2885    1.05  0.29565    
    ## sp8 X3            -1.0177  0.2674   -3.81  0.00014 ***
    ## sp9 (Intercept)    0.0251  0.1699    0.15  0.88257    
    ## sp9 X1             1.0902  0.3260    3.34  0.00083 ***
    ## sp9 X2            -0.8162  0.3227   -2.53  0.01143 *  
    ## sp9 X3             0.6126  0.2842    2.16  0.03111 *  
    ## sp10 (Intercept)  -0.0960  0.1874   -0.51  0.60858    
    ## sp10 X1           -0.4447  0.3512   -1.27  0.20544    
    ## sp10 X2           -0.9639  0.3397   -2.84  0.00454 ** 
    ## sp10 X3           -0.3804  0.3216   -1.18  0.23678    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
plot(model)
```

    ## LogLik:  -511.7972 
    ## Regularization loss:  0 
    ## 
    ## Species-species correlation matrix: 
    ## 
    ##  sp1  1.0000                                 
    ##  sp2 -0.3560  1.0000                             
    ##  sp3 -0.1180 -0.3740  1.0000                         
    ##  sp4 -0.1220 -0.3350  0.6950  1.0000                     
    ##  sp5  0.5970 -0.3510 -0.0760 -0.0870  1.0000                 
    ##  sp6 -0.2140  0.3880  0.1420  0.1350 -0.0730  1.0000             
    ##  sp7  0.4640 -0.1540  0.1470  0.1390  0.5120  0.2440  1.0000         
    ##  sp8  0.2560  0.1570 -0.4240 -0.4180  0.2150 -0.0320  0.1070  1.0000     
    ##  sp9 -0.0730 -0.0470  0.0540  0.0700 -0.3050 -0.2550 -0.2220 -0.1010  1.0000 
    ##  sp10     0.1800  0.3920 -0.6130 -0.5760  0.2220  0.1000  0.1090  0.4120 -0.2190  1.0000
    ## 
    ## 
    ## 
    ## Spatial: 
    ##             sp1        sp2       sp3        sp4       sp5       sp6       sp7
    ## X1:X2 0.2260564 -0.5308802 0.5423007 0.03226206 0.3345731 0.1966482 0.4499481
    ##             sp8       sp9      sp10
    ## X1:X2 0.4780776 0.2826627 0.2449929
    ## 
    ## 
    ## 
    ##                  Estimate Std.Err Z value Pr(>|z|)    
    ## sp1 (Intercept)   -0.0290  0.2081   -0.14  0.88925    
    ## sp1 X1             0.8329  0.3936    2.12  0.03433 *  
    ## sp1 X2            -1.6153  0.3787   -4.27  2.0e-05 ***
    ## sp1 X3            -0.1006  0.3489   -0.29  0.77308    
    ## sp2 (Intercept)   -0.0335  0.2164   -0.16  0.87681    
    ## sp2 X1             0.8809  0.4079    2.16  0.03080 *  
    ## sp2 X2             0.1921  0.3994    0.48  0.63055    
    ## sp2 X3             0.4257  0.3576    1.19  0.23383    
    ## sp3 (Intercept)   -0.3370  0.2195   -1.54  0.12472    
    ## sp3 X1             1.0309  0.4044    2.55  0.01080 *  
    ## sp3 X2            -0.2759  0.3924   -0.70  0.48190    
    ## sp3 X3            -0.7200  0.3764   -1.91  0.05577 .  
    ## sp4 (Intercept)   -0.0535  0.1962   -0.27  0.78510    
    ## sp4 X1            -1.0306  0.3724   -2.77  0.00565 ** 
    ## sp4 X2            -1.2974  0.3783   -3.43  0.00061 ***
    ## sp4 X3            -0.3160  0.3288   -0.96  0.33662    
    ## sp5 (Intercept)   -0.1503  0.2044   -0.74  0.46211    
    ## sp5 X1             0.5085  0.3914    1.30  0.19383    
    ## sp5 X2             0.4517  0.3673    1.23  0.21880    
    ## sp5 X3            -0.4063  0.3406   -1.19  0.23291    
    ## sp6 (Intercept)    0.1879  0.2058    0.91  0.36116    
    ## sp6 X1             1.6793  0.4291    3.91  9.1e-05 ***
    ## sp6 X2            -0.6698  0.3940   -1.70  0.08912 .  
    ## sp6 X3             0.1595  0.3369    0.47  0.63595    
    ## sp7 (Intercept)   -0.0188  0.2002   -0.09  0.92511    
    ## sp7 X1            -0.2593  0.3889   -0.67  0.50501    
    ## sp7 X2             0.3116  0.3679    0.85  0.39698    
    ## sp7 X3            -1.0443  0.3462   -3.02  0.00255 ** 
    ## sp8 (Intercept)    0.1435  0.1533    0.94  0.34911    
    ## sp8 X1             0.2595  0.2950    0.88  0.37895    
    ## sp8 X2             0.3017  0.2885    1.05  0.29565    
    ## sp8 X3            -1.0177  0.2674   -3.81  0.00014 ***
    ## sp9 (Intercept)    0.0251  0.1699    0.15  0.88257    
    ## sp9 X1             1.0902  0.3260    3.34  0.00083 ***
    ## sp9 X2            -0.8162  0.3227   -2.53  0.01143 *  
    ## sp9 X3             0.6126  0.2842    2.16  0.03111 *  
    ## sp10 (Intercept)  -0.0960  0.1874   -0.51  0.60858    
    ## sp10 X1           -0.4447  0.3512   -1.27  0.20544    
    ## sp10 X2           -0.9639  0.3397   -2.84  0.00454 ** 
    ## sp10 X3           -0.3804  0.3216   -1.18  0.23678    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

![](README_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

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
print(imp)
```

    ##    sp       env      spatial     biotic
    ## 1   1 0.8509671 3.718882e-04 0.14866112
    ## 2   2 0.5491779 3.699944e-03 0.44712225
    ## 3   3 0.7280060 3.747982e-03 0.26824600
    ## 4   4 0.9064414 7.932865e-06 0.09355063
    ## 5   5 0.4953244 2.304945e-03 0.50237066
    ## 6   6 0.8561825 2.861650e-04 0.14353135
    ## 7   7 0.5219943 1.883551e-03 0.47612211
    ## 8   8 0.9194061 4.193856e-03 0.07640003
    ## 9   9 0.7399929 6.984830e-04 0.25930864
    ## 10 10 0.9448155 1.054667e-03 0.05412992

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
    ## Biotic  150.78920         521.71684       0.77862      0.1091
    ## Abiotic 185.05100         336.66584       0.96521      0.2429
    ## Spatial   2.61524         334.05060       0.96611      0.2448

``` r
plot(an)
```

![](README_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

The anova shows the relative changes in the deviance of the groups and
their intersections.

We can also visualize the individual contributions to the species and
site

![R^2](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;R%5E2 "R^2")

:

``` r
plot(an, internal=TRUE)
```

    ## Registered S3 methods overwritten by 'ggtern':
    ##   method           from   
    ##   grid.draw.ggplot ggplot2
    ##   plot.ggplot      ggplot2
    ##   print.ggplot     ggplot2

![](README_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

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

    ## Iter: 0/10   0%|          | [00:00, ?it/s]Iter: 0/10   0%|          | [00:00, ?it/s, loss=7.328]Iter: 1/10  10%|#         | [00:00,  5.28it/s, loss=7.328]Iter: 1/10  10%|#         | [00:00,  5.28it/s, loss=7.303]Iter: 1/10  10%|#         | [00:00,  5.28it/s, loss=7.305]Iter: 1/10  10%|#         | [00:00,  5.28it/s, loss=7.328]Iter: 1/10  10%|#         | [00:00,  5.28it/s, loss=7.304]Iter: 1/10  10%|#         | [00:00,  5.28it/s, loss=7.292]Iter: 6/10  60%|######    | [00:00, 23.82it/s, loss=7.292]Iter: 6/10  60%|######    | [00:00, 23.82it/s, loss=7.305]Iter: 6/10  60%|######    | [00:00, 23.82it/s, loss=7.285]Iter: 6/10  60%|######    | [00:00, 23.82it/s, loss=7.275]Iter: 6/10  60%|######    | [00:00, 23.82it/s, loss=7.261]Iter: 10/10 100%|##########| [00:00, 26.27it/s, loss=7.261]

Calculate Importance:

``` python
Beta = np.transpose(model.env_weights[0])
Sigma = ( model.sigma @ model.sigma.t() + torch.diag(torch.ones([1])) ).data.cpu().numpy()
covX = fa.covariance( torch.tensor(Env).t() ).data.cpu().numpy()

fa.importance(beta=Beta, covX=covX, sigma=Sigma)
```

    ## {'env': array([[ 3.7089107e-03,  6.4757657e-03,  2.1139428e-03,  7.8506982e-03,
    ##          5.9689581e-03],
    ##        [ 1.6079791e-02,  3.6896379e-03,  5.2381670e-03,  6.0370858e-03,
    ##          1.1554083e-03],
    ##        [ 6.1437520e-03,  9.7632296e-03,  7.4961684e-03, -3.9039826e-04,
    ##          7.9328282e-04],
    ##        [ 9.9422687e-05,  5.3525879e-04,  1.6309794e-03,  1.7054105e-02,
    ##          1.8254045e-02],
    ##        [ 6.3455396e-04,  9.6351805e-04,  9.7591011e-04,  1.1000523e-04,
    ##          3.6621885e-03],
    ##        [ 1.5674103e-02,  1.0940240e-03,  1.0806777e-02,  8.0627156e-03,
    ##          2.0850746e-02],
    ##        [ 1.2024483e-05,  5.0663785e-03,  1.4675152e-02,  1.2784588e-02,
    ##          5.2863178e-03],
    ##        [ 2.9168164e-03,  2.0089261e-04,  1.3217300e-02,  4.6121329e-04,
    ##          1.9982974e-05],
    ##        [ 1.0323757e-02,  2.7552131e-03,  8.2153408e-04,  6.8914989e-04,
    ##          5.9855417e-03],
    ##        [ 7.2189951e-03,  4.6700318e-03,  4.4443756e-03,  2.7086365e-03,
    ##          2.4578572e-05]], dtype=float32), 'biotic': array([0.9738818 , 0.96779996, 0.9761939 , 0.9624262 , 0.9936538 ,
    ##        0.94351166, 0.9621755 , 0.9831838 , 0.97942483, 0.9809334 ],
    ##       dtype=float32)}
