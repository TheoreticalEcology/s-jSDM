
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
\>= 3.7 and pytorch must be installed (more details below).

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
devtools::install_github("https://github.com/TheoreticalEcology/s-jSDM", subdir = "sjSDM", ref = "master")
```

Once the dependencies are installed, the following code should run:

## Workflow

Simulate a community and fit a sjSDM model:

``` r
library(sjSDM)
```

    ## ── Attaching sjSDM ──────────────────────────────────────────────────── 1.0.4 ──

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

    ## Family:  binomial 
    ## 
    ## LogLik:  -507.6378 
    ## Regularization loss:  0 
    ## 
    ## Species-species correlation matrix: 
    ## 
    ##  sp1  1.0000                                 
    ##  sp2 -0.3810  1.0000                             
    ##  sp3 -0.2130 -0.3990  1.0000                         
    ##  sp4 -0.1840 -0.3690  0.8270  1.0000                     
    ##  sp5  0.6710 -0.3760 -0.1570 -0.1310  1.0000                 
    ##  sp6 -0.3370  0.4970  0.1760  0.1840 -0.1420  1.0000             
    ##  sp7  0.5690 -0.1190  0.1060  0.1440  0.5380  0.2410  1.0000         
    ##  sp8  0.3050  0.1830 -0.5110 -0.5070  0.2370 -0.0640  0.1400  1.0000     
    ##  sp9 -0.0600 -0.0300  0.0700  0.0780 -0.4100 -0.3240 -0.2290 -0.1160  1.0000 
    ##  sp10     0.2060  0.4760 -0.7200 -0.6770  0.2640  0.1220  0.1280  0.4630 -0.2670  1.0000
    ## 
    ## 
    ## 
    ## Spatial: 
    ##            sp1       sp2      sp3       sp4      sp5      sp6      sp7      sp8
    ## X1:X2 2.097189 -4.265176 3.183704 0.4749841 2.917459 1.054281 3.151275 1.926711
    ##             sp9     sp10
    ## X1:X2 0.9519854 1.382612
    ## 
    ## 
    ## 
    ##                  Estimate  Std.Err Z value Pr(>|z|)    
    ## sp1 (Intercept)  -0.04250  0.26957   -0.16  0.87472    
    ## sp1 X1            1.35201  0.51034    2.65  0.00807 ** 
    ## sp1 X2           -2.42558  0.50479   -4.81  1.5e-06 ***
    ## sp1 X3           -0.26812  0.44301   -0.61  0.54502    
    ## sp2 (Intercept)  -0.02252  0.25054   -0.09  0.92837    
    ## sp2 X1            1.29599  0.45133    2.87  0.00409 ** 
    ## sp2 X2            0.27695  0.46789    0.59  0.55390    
    ## sp2 X3            0.61270  0.41997    1.46  0.14459    
    ## sp3 (Intercept)  -0.50246  0.27216   -1.85  0.06486 .  
    ## sp3 X1            1.49316  0.50053    2.98  0.00285 ** 
    ## sp3 X2           -0.49962  0.47866   -1.04  0.29658    
    ## sp3 X3           -1.02865  0.45183   -2.28  0.02281 *  
    ## sp4 (Intercept)  -0.07307  0.24350   -0.30  0.76409    
    ## sp4 X1           -1.52630  0.50579   -3.02  0.00255 ** 
    ## sp4 X2           -1.96613  0.47429   -4.15  3.4e-05 ***
    ## sp4 X3           -0.37335  0.40726   -0.92  0.35928    
    ## sp5 (Intercept)  -0.20108  0.25579   -0.79  0.43179    
    ## sp5 X1            0.74516  0.51170    1.46  0.14532    
    ## sp5 X2            0.57111  0.48656    1.17  0.24049    
    ## sp5 X3           -0.72480  0.44636   -1.62  0.10442    
    ## sp6 (Intercept)   0.24978  0.25516    0.98  0.32762    
    ## sp6 X1            2.51313  0.53761    4.67  2.9e-06 ***
    ## sp6 X2           -1.07828  0.49344   -2.19  0.02887 *  
    ## sp6 X3            0.11864  0.40557    0.29  0.76989    
    ## sp7 (Intercept)  -0.00135  0.27085    0.00  0.99602    
    ## sp7 X1           -0.33149  0.54594   -0.61  0.54373    
    ## sp7 X2            0.32389  0.48590    0.67  0.50504    
    ## sp7 X3           -1.56980  0.45237   -3.47  0.00052 ***
    ## sp8 (Intercept)   0.13244  0.15898    0.83  0.40481    
    ## sp8 X1            0.30743  0.30592    1.00  0.31493    
    ## sp8 X2            0.30791  0.29553    1.04  0.29746    
    ## sp8 X3           -1.17697  0.28212   -4.17  3.0e-05 ***
    ## sp9 (Intercept)   0.02497  0.19789    0.13  0.89958    
    ## sp9 X1            1.38997  0.36889    3.77  0.00016 ***
    ## sp9 X2           -1.07082  0.36722   -2.92  0.00355 ** 
    ## sp9 X3            0.80624  0.32852    2.45  0.01412 *  
    ## sp10 (Intercept) -0.13245  0.20378   -0.65  0.51572    
    ## sp10 X1          -0.54231  0.39152   -1.39  0.16601    
    ## sp10 X2          -1.26720  0.36897   -3.43  0.00059 ***
    ## sp10 X3          -0.60400  0.34778   -1.74  0.08244 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
plot(model)
```

    ## Family:  binomial 
    ## 
    ## LogLik:  -507.6378 
    ## Regularization loss:  0 
    ## 
    ## Species-species correlation matrix: 
    ## 
    ##  sp1  1.0000                                 
    ##  sp2 -0.3810  1.0000                             
    ##  sp3 -0.2130 -0.3990  1.0000                         
    ##  sp4 -0.1840 -0.3690  0.8270  1.0000                     
    ##  sp5  0.6710 -0.3760 -0.1570 -0.1310  1.0000                 
    ##  sp6 -0.3370  0.4970  0.1760  0.1840 -0.1420  1.0000             
    ##  sp7  0.5690 -0.1190  0.1060  0.1440  0.5380  0.2410  1.0000         
    ##  sp8  0.3050  0.1830 -0.5110 -0.5070  0.2370 -0.0640  0.1400  1.0000     
    ##  sp9 -0.0600 -0.0300  0.0700  0.0780 -0.4100 -0.3240 -0.2290 -0.1160  1.0000 
    ##  sp10     0.2060  0.4760 -0.7200 -0.6770  0.2640  0.1220  0.1280  0.4630 -0.2670  1.0000
    ## 
    ## 
    ## 
    ## Spatial: 
    ##            sp1       sp2      sp3       sp4      sp5      sp6      sp7      sp8
    ## X1:X2 2.097189 -4.265176 3.183704 0.4749841 2.917459 1.054281 3.151275 1.926711
    ##             sp9     sp10
    ## X1:X2 0.9519854 1.382612
    ## 
    ## 
    ## 
    ##                  Estimate  Std.Err Z value Pr(>|z|)    
    ## sp1 (Intercept)  -0.04250  0.26957   -0.16  0.87472    
    ## sp1 X1            1.35201  0.51034    2.65  0.00807 ** 
    ## sp1 X2           -2.42558  0.50479   -4.81  1.5e-06 ***
    ## sp1 X3           -0.26812  0.44301   -0.61  0.54502    
    ## sp2 (Intercept)  -0.02252  0.25054   -0.09  0.92837    
    ## sp2 X1            1.29599  0.45133    2.87  0.00409 ** 
    ## sp2 X2            0.27695  0.46789    0.59  0.55390    
    ## sp2 X3            0.61270  0.41997    1.46  0.14459    
    ## sp3 (Intercept)  -0.50246  0.27216   -1.85  0.06486 .  
    ## sp3 X1            1.49316  0.50053    2.98  0.00285 ** 
    ## sp3 X2           -0.49962  0.47866   -1.04  0.29658    
    ## sp3 X3           -1.02865  0.45183   -2.28  0.02281 *  
    ## sp4 (Intercept)  -0.07307  0.24350   -0.30  0.76409    
    ## sp4 X1           -1.52630  0.50579   -3.02  0.00255 ** 
    ## sp4 X2           -1.96613  0.47429   -4.15  3.4e-05 ***
    ## sp4 X3           -0.37335  0.40726   -0.92  0.35928    
    ## sp5 (Intercept)  -0.20108  0.25579   -0.79  0.43179    
    ## sp5 X1            0.74516  0.51170    1.46  0.14532    
    ## sp5 X2            0.57111  0.48656    1.17  0.24049    
    ## sp5 X3           -0.72480  0.44636   -1.62  0.10442    
    ## sp6 (Intercept)   0.24978  0.25516    0.98  0.32762    
    ## sp6 X1            2.51313  0.53761    4.67  2.9e-06 ***
    ## sp6 X2           -1.07828  0.49344   -2.19  0.02887 *  
    ## sp6 X3            0.11864  0.40557    0.29  0.76989    
    ## sp7 (Intercept)  -0.00135  0.27085    0.00  0.99602    
    ## sp7 X1           -0.33149  0.54594   -0.61  0.54373    
    ## sp7 X2            0.32389  0.48590    0.67  0.50504    
    ## sp7 X3           -1.56980  0.45237   -3.47  0.00052 ***
    ## sp8 (Intercept)   0.13244  0.15898    0.83  0.40481    
    ## sp8 X1            0.30743  0.30592    1.00  0.31493    
    ## sp8 X2            0.30791  0.29553    1.04  0.29746    
    ## sp8 X3           -1.17697  0.28212   -4.17  3.0e-05 ***
    ## sp9 (Intercept)   0.02497  0.19789    0.13  0.89958    
    ## sp9 X1            1.38997  0.36889    3.77  0.00016 ***
    ## sp9 X2           -1.07082  0.36722   -2.92  0.00355 ** 
    ## sp9 X3            0.80624  0.32852    2.45  0.01412 *  
    ## sp10 (Intercept) -0.13245  0.20378   -0.65  0.51572    
    ## sp10 X1          -0.54231  0.39152   -1.39  0.16601    
    ## sp10 X2          -1.26720  0.36897   -3.43  0.00059 ***
    ## sp10 X3          -0.60400  0.34778   -1.74  0.08244 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

![](README_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

We support other distributions:

- Count data with Poisson:

``` r
model <- sjSDM(Y = Occ, env = linear(data = Env, formula = ~X1+X2+X3), spatial = linear(data = SP, formula = ~0+X1:X2), se = TRUE, family=poisson("log"))
```

- Count data with negative Binomial (which is still experimental, if you
  run into errors/problems, please let us know):

``` r
model <- sjSDM(Y = Occ, env = linear(data = Env, formula = ~X1+X2+X3), spatial = linear(data = SP, formula = ~0+X1:X2), se = TR, family="nbinom")
```

- Gaussian (normal):

``` r
model <- sjSDM(Y = Occ, env = linear(data = Env, formula = ~X1+X2+X3), spatial = linear(data = SP, formula = ~0+X1:X2), se = TR, family=gaussian("identity"))
```

### Anova

ANOVA can be used to partition the three components (abiotic, biotic,
and spatial):

``` r
an = anova(model)
print(an)
```

    ## Analysis of Deviance Table
    ## 
    ## Terms added sequentially:
    ## 
    ##           Deviance Residual deviance R2 Nagelkerke R2 McFadden
    ## Abiotic  159.06804        1176.20888       0.79621      0.1147
    ## Biotic   176.95395        1158.32297       0.82959      0.1276
    ## Spatial   14.27360        1321.00332       0.13302      0.0103
    ## Full     383.95964         951.31728       0.97850      0.2770

``` r
plot(an)
```

![](README_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

The anova shows the relative changes in the R<sup>2</sup> of the groups
and their intersections.

### Internal metacommunity structure

Following [Leibold et al., 2022](https://doi.org/10.1111/oik.08618) we
can calculate and visualize the internal metacommunity structure
(=partitioning of the three components for species and sites). The
internal structure is already calculated by the ANOVA and we can
visualize it with the plot method:

``` r
results = plotInternalStructure(an) # or plot(an, internal = TRUE)
```

    ## Registered S3 methods overwritten by 'ggtern':
    ##   method           from   
    ##   grid.draw.ggplot ggplot2
    ##   plot.ggplot      ggplot2
    ##   print.ggplot     ggplot2

![](README_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

The plot function returns the results for the internal metacommunity
structure:

``` r
print(results$data$Species)
```

    ##           env          spa     codist         r2
    ## 1  0.17385464 0.0179602294 0.16186165 0.03536765
    ## 2  0.08866249 0.0000000000 0.18919762 0.02774472
    ## 3  0.12583128 0.0173940956 0.20101050 0.03442359
    ## 4  0.16422076 0.0120107958 0.15563226 0.03318638
    ## 5  0.08398346 0.0049290259 0.16448047 0.02533930
    ## 6  0.18501795 0.0009280466 0.11474444 0.03006904
    ## 7  0.11573965 0.0000000000 0.13868028 0.02522951
    ## 8  0.13613331 0.0067601632 0.05508682 0.01979803
    ## 9  0.17587926 0.0000000000 0.04784122 0.02220423
    ## 10 0.09485239 0.0000000000 0.14503868 0.02360589

## Installation trouble shooting

If the installation fails, check out the help of ?install_sjSDM,
?installation_help, and vignette(“Dependencies”, package = “sjSDM”).

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

    ## Iter: 0/10   0%|          | [00:00, ?it/s]Iter: 0/10   0%|          | [00:00, ?it/s, loss=7.287]Iter: 1/10  10%|#         | [00:00,  3.93it/s, loss=7.287]Iter: 1/10  10%|#         | [00:00,  3.93it/s, loss=7.274]Iter: 1/10  10%|#         | [00:00,  3.93it/s, loss=7.29] Iter: 1/10  10%|#         | [00:00,  3.93it/s, loss=7.268]Iter: 1/10  10%|#         | [00:00,  3.93it/s, loss=7.239]Iter: 1/10  10%|#         | [00:00,  3.93it/s, loss=7.246]Iter: 1/10  10%|#         | [00:00,  3.93it/s, loss=7.264]Iter: 1/10  10%|#         | [00:00,  3.93it/s, loss=7.252]Iter: 1/10  10%|#         | [00:00,  3.93it/s, loss=7.249]Iter: 1/10  10%|#         | [00:00,  3.93it/s, loss=7.252]Iter: 10/10 100%|##########| [00:00, 33.40it/s, loss=7.252]Iter: 10/10 100%|##########| [00:00, 27.25it/s, loss=7.252]

Calculate Importance:

``` python
Beta = np.transpose(model.env_weights[0])
Sigma = ( model.sigma @ model.sigma.t() + torch.diag(torch.ones([1])) ).data.cpu().numpy()
covX = fa.covariance( torch.tensor(Env).t() ).data.cpu().numpy()

fa.importance(beta=Beta, covX=covX, sigma=Sigma)
```

    ## {'env': array([[ 9.21158865e-03,  3.65287095e-04,  6.48367032e-03,
    ##          6.89318869e-04,  9.29490384e-03],
    ##        [ 1.77851284e-03,  1.10952486e-03,  5.15174586e-03,
    ##          2.44008703e-03,  9.79916751e-03],
    ##        [-4.03536193e-04,  1.67346653e-02,  6.27224683e-04,
    ##          1.85741577e-02,  6.20362302e-03],
    ##        [ 3.37869744e-03,  3.97691550e-03,  5.26852149e-04,
    ##          1.31604644e-02,  9.52953100e-03],
    ##        [ 7.39491777e-03,  1.12318620e-02,  6.52327295e-03,
    ##          3.26113706e-03,  4.22334019e-03],
    ##        [ 1.36883755e-03,  1.90022290e-02,  1.41613511e-02,
    ##          1.13295615e-02,  8.60763481e-04],
    ##        [ 4.00130078e-03,  1.83231881e-04,  1.62727386e-02,
    ##          3.67835839e-03,  1.23462146e-02],
    ##        [ 2.25505233e-03,  8.65742378e-03,  2.98037822e-03,
    ##          1.16588371e-02,  2.06654775e-03],
    ##        [ 5.18416427e-03,  1.95248995e-03,  5.31449215e-04,
    ##         -2.36918004e-05,  5.39716857e-04],
    ##        [ 3.95786343e-03,  1.45690329e-02,  2.24688882e-03,
    ##          1.94203306e-03,  2.39070994e-03]], dtype=float32), 'biotic': array([0.9739553 , 0.9797209 , 0.9582639 , 0.9694275 , 0.96736544,
    ##        0.9532773 , 0.9635182 , 0.9723818 , 0.9918159 , 0.9748935 ],
    ##       dtype=float32)}
