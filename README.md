
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

    ## LogLik:  -519.8094 
    ## Regularization loss:  0 
    ## 
    ## Species-species correlation matrix: 
    ## 
    ##  sp1  1.0000                                 
    ##  sp2 -0.3440  1.0000                             
    ##  sp3 -0.1390 -0.3690  1.0000                         
    ##  sp4 -0.1370 -0.3390  0.7050  1.0000                     
    ##  sp5  0.5950 -0.3340 -0.0860 -0.0830  1.0000                 
    ##  sp6 -0.2220  0.3890  0.1380  0.1480 -0.0650  1.0000             
    ##  sp7  0.4960 -0.1640  0.1270  0.1280  0.5100  0.2070  1.0000         
    ##  sp8  0.2690  0.1490 -0.4330 -0.4120  0.2310 -0.0470  0.1300  1.0000     
    ##  sp9 -0.0840 -0.0560  0.0780  0.0680 -0.3270 -0.2490 -0.1960 -0.1040  1.0000 
    ##  sp10     0.1990  0.3880 -0.5890 -0.5550  0.2370  0.1170  0.1320  0.4050 -0.2450  1.0000
    ## 
    ## 
    ## 
    ## Spatial: 
    ##             sp1        sp2       sp3          sp4       sp5       sp6       sp7
    ## X1:X2 0.2177128 -0.5440111 0.5465564 -0.001427716 0.3573959 0.2163743 0.4391951
    ##             sp8      sp9      sp10
    ## X1:X2 0.4889981 0.265004 0.2422081
    ## 
    ## 
    ## 
    ##                   Estimate   Std.Err Z value Pr(>|z|)    
    ## sp1 (Intercept)  -0.029840  0.217539   -0.14  0.89089    
    ## sp1 X1            0.825156  0.400700    2.06  0.03947 *  
    ## sp1 X2           -1.625258  0.390897   -4.16  3.2e-05 ***
    ## sp1 X3           -0.127307  0.362164   -0.35  0.72520    
    ## sp2 (Intercept)  -0.022792  0.218462   -0.10  0.91691    
    ## sp2 X1            0.869993  0.425023    2.05  0.04066 *  
    ## sp2 X2            0.146673  0.404662    0.36  0.71701    
    ## sp2 X3            0.409775  0.365122    1.12  0.26174    
    ## sp3 (Intercept)  -0.360172  0.220931   -1.63  0.10305    
    ## sp3 X1            1.048723  0.391881    2.68  0.00745 ** 
    ## sp3 X2           -0.243140  0.400844   -0.61  0.54414    
    ## sp3 X3           -0.693035  0.383769   -1.81  0.07094 .  
    ## sp4 (Intercept)  -0.051956  0.201594   -0.26  0.79662    
    ## sp4 X1           -1.016854  0.380479   -2.67  0.00753 ** 
    ## sp4 X2           -1.298405  0.379201   -3.42  0.00062 ***
    ## sp4 X3           -0.274420  0.348587   -0.79  0.43114    
    ## sp5 (Intercept)  -0.142492  0.206063   -0.69  0.48925    
    ## sp5 X1            0.477353  0.379789    1.26  0.20879    
    ## sp5 X2            0.418336  0.373346    1.12  0.26250    
    ## sp5 X3           -0.446934  0.352626   -1.27  0.20500    
    ## sp6 (Intercept)   0.180630  0.201184    0.90  0.36927    
    ## sp6 X1            1.693508  0.415504    4.08  4.6e-05 ***
    ## sp6 X2           -0.691093  0.386790   -1.79  0.07398 .  
    ## sp6 X3            0.151210  0.337839    0.45  0.65446    
    ## sp7 (Intercept)  -0.000269  0.204736    0.00  0.99895    
    ## sp7 X1           -0.301724  0.399147   -0.76  0.44970    
    ## sp7 X2            0.293641  0.372641    0.79  0.43070    
    ## sp7 X3           -1.089956  0.342880   -3.18  0.00148 ** 
    ## sp8 (Intercept)   0.144771  0.150699    0.96  0.33672    
    ## sp8 X1            0.243090  0.289498    0.84  0.40108    
    ## sp8 X2            0.287361  0.282389    1.02  0.30886    
    ## sp8 X3           -1.039104  0.264258   -3.93  8.4e-05 ***
    ## sp9 (Intercept)   0.031095  0.171879    0.18  0.85644    
    ## sp9 X1            1.099648  0.330783    3.32  0.00089 ***
    ## sp9 X2           -0.790673  0.324681   -2.44  0.01488 *  
    ## sp9 X3            0.627002  0.288140    2.18  0.02955 *  
    ## sp10 (Intercept) -0.079505  0.179561   -0.44  0.65793    
    ## sp10 X1          -0.461844  0.336563   -1.37  0.16999    
    ## sp10 X2          -0.981453  0.331483   -2.96  0.00307 ** 
    ## sp10 X3          -0.394081  0.301172   -1.31  0.19071    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
plot(model)
```

    ## LogLik:  -519.8094 
    ## Regularization loss:  0 
    ## 
    ## Species-species correlation matrix: 
    ## 
    ##  sp1  1.0000                                 
    ##  sp2 -0.3440  1.0000                             
    ##  sp3 -0.1390 -0.3690  1.0000                         
    ##  sp4 -0.1370 -0.3390  0.7050  1.0000                     
    ##  sp5  0.5950 -0.3340 -0.0860 -0.0830  1.0000                 
    ##  sp6 -0.2220  0.3890  0.1380  0.1480 -0.0650  1.0000             
    ##  sp7  0.4960 -0.1640  0.1270  0.1280  0.5100  0.2070  1.0000         
    ##  sp8  0.2690  0.1490 -0.4330 -0.4120  0.2310 -0.0470  0.1300  1.0000     
    ##  sp9 -0.0840 -0.0560  0.0780  0.0680 -0.3270 -0.2490 -0.1960 -0.1040  1.0000 
    ##  sp10     0.1990  0.3880 -0.5890 -0.5550  0.2370  0.1170  0.1320  0.4050 -0.2450  1.0000
    ## 
    ## 
    ## 
    ## Spatial: 
    ##             sp1        sp2       sp3          sp4       sp5       sp6       sp7
    ## X1:X2 0.2177128 -0.5440111 0.5465564 -0.001427716 0.3573959 0.2163743 0.4391951
    ##             sp8      sp9      sp10
    ## X1:X2 0.4889981 0.265004 0.2422081
    ## 
    ## 
    ## 
    ##                   Estimate   Std.Err Z value Pr(>|z|)    
    ## sp1 (Intercept)  -0.029840  0.217539   -0.14  0.89089    
    ## sp1 X1            0.825156  0.400700    2.06  0.03947 *  
    ## sp1 X2           -1.625258  0.390897   -4.16  3.2e-05 ***
    ## sp1 X3           -0.127307  0.362164   -0.35  0.72520    
    ## sp2 (Intercept)  -0.022792  0.218462   -0.10  0.91691    
    ## sp2 X1            0.869993  0.425023    2.05  0.04066 *  
    ## sp2 X2            0.146673  0.404662    0.36  0.71701    
    ## sp2 X3            0.409775  0.365122    1.12  0.26174    
    ## sp3 (Intercept)  -0.360172  0.220931   -1.63  0.10305    
    ## sp3 X1            1.048723  0.391881    2.68  0.00745 ** 
    ## sp3 X2           -0.243140  0.400844   -0.61  0.54414    
    ## sp3 X3           -0.693035  0.383769   -1.81  0.07094 .  
    ## sp4 (Intercept)  -0.051956  0.201594   -0.26  0.79662    
    ## sp4 X1           -1.016854  0.380479   -2.67  0.00753 ** 
    ## sp4 X2           -1.298405  0.379201   -3.42  0.00062 ***
    ## sp4 X3           -0.274420  0.348587   -0.79  0.43114    
    ## sp5 (Intercept)  -0.142492  0.206063   -0.69  0.48925    
    ## sp5 X1            0.477353  0.379789    1.26  0.20879    
    ## sp5 X2            0.418336  0.373346    1.12  0.26250    
    ## sp5 X3           -0.446934  0.352626   -1.27  0.20500    
    ## sp6 (Intercept)   0.180630  0.201184    0.90  0.36927    
    ## sp6 X1            1.693508  0.415504    4.08  4.6e-05 ***
    ## sp6 X2           -0.691093  0.386790   -1.79  0.07398 .  
    ## sp6 X3            0.151210  0.337839    0.45  0.65446    
    ## sp7 (Intercept)  -0.000269  0.204736    0.00  0.99895    
    ## sp7 X1           -0.301724  0.399147   -0.76  0.44970    
    ## sp7 X2            0.293641  0.372641    0.79  0.43070    
    ## sp7 X3           -1.089956  0.342880   -3.18  0.00148 ** 
    ## sp8 (Intercept)   0.144771  0.150699    0.96  0.33672    
    ## sp8 X1            0.243090  0.289498    0.84  0.40108    
    ## sp8 X2            0.287361  0.282389    1.02  0.30886    
    ## sp8 X3           -1.039104  0.264258   -3.93  8.4e-05 ***
    ## sp9 (Intercept)   0.031095  0.171879    0.18  0.85644    
    ## sp9 X1            1.099648  0.330783    3.32  0.00089 ***
    ## sp9 X2           -0.790673  0.324681   -2.44  0.01488 *  
    ## sp9 X3            0.627002  0.288140    2.18  0.02955 *  
    ## sp10 (Intercept) -0.079505  0.179561   -0.44  0.65793    
    ## sp10 X1          -0.461844  0.336563   -1.37  0.16999    
    ## sp10 X2          -0.981453  0.331483   -2.96  0.00307 ** 
    ## sp10 X3          -0.394081  0.301172   -1.31  0.19071    
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
print(imp)
```

    ##    sp       env      spatial      biotic
    ## 1   1 0.8489780 3.413156e-04 0.150680736
    ## 2   2 0.5341426 3.999921e-03 0.461857498
    ## 3   3 0.7394266 3.917745e-03 0.256655753
    ## 4   4 0.9132825 1.599796e-08 0.086717583
    ## 5   5 0.4769205 2.633283e-03 0.520446241
    ## 6   6 0.8629941 3.422207e-04 0.136663601
    ## 7   7 0.5407141 1.700569e-03 0.457585305
    ## 8   8 0.9208101 4.275649e-03 0.074914329
    ## 9   9 0.7375954 6.099563e-04 0.261794657
    ## 10 10 0.9931872 1.033048e-03 0.005779827

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
    ## Biotic  156.82986         515.65547       0.79160      0.1134
    ## Abiotic 177.42598         338.22950       0.96465      0.2418
    ## Spatial  11.86151         326.36799       0.96861      0.2504

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

    ## Iter: 0/10   0%|          | [00:00, ?it/s]Iter: 0/10   0%|          | [00:00, ?it/s, loss=7.336]Iter: 1/10  10%|#         | [00:00,  5.86it/s, loss=7.336]Iter: 1/10  10%|#         | [00:00,  5.86it/s, loss=7.34] Iter: 1/10  10%|#         | [00:00,  5.86it/s, loss=7.312]Iter: 1/10  10%|#         | [00:00,  5.86it/s, loss=7.318]Iter: 1/10  10%|#         | [00:00,  5.86it/s, loss=7.324]Iter: 1/10  10%|#         | [00:00,  5.86it/s, loss=7.331]Iter: 1/10  10%|#         | [00:00,  5.86it/s, loss=7.318]Iter: 7/10  70%|#######   | [00:00, 28.58it/s, loss=7.318]Iter: 7/10  70%|#######   | [00:00, 28.58it/s, loss=7.318]Iter: 7/10  70%|#######   | [00:00, 28.58it/s, loss=7.321]Iter: 7/10  70%|#######   | [00:00, 28.58it/s, loss=7.306]Iter: 10/10 100%|##########| [00:00, 29.75it/s, loss=7.306]

Calculate Importance:

``` python
Beta = np.transpose(model.env_weights[0])
Sigma = ( model.sigma @ model.sigma.t() + torch.diag(torch.ones([1])) ).data.cpu().numpy()
covX = fa.covariance( torch.tensor(Env).t() ).data.cpu().numpy()

fa.importance(beta=Beta, covX=covX, sigma=Sigma)
```

    ## {'env': array([[ 1.87528494e-03,  1.33683877e-02,  3.21150525e-03,
    ##          2.23890209e-04,  1.15812254e-04],
    ##        [ 1.59566756e-04,  1.85161713e-04,  1.18317269e-02,
    ##          4.80526433e-06,  2.01872666e-03],
    ##        [ 1.68252829e-02,  1.10603614e-04,  5.71511080e-03,
    ##          7.22679123e-03,  1.42804789e-03],
    ##        [ 9.11659808e-05,  2.42104139e-02,  1.59541913e-03,
    ##          5.71383862e-03,  2.00973544e-03],
    ##        [ 3.34012485e-03,  1.53095066e-03,  9.65126674e-04,
    ##          2.07257946e-03, -4.77658468e-05],
    ##        [ 9.48124379e-03,  1.16053578e-02,  2.46209279e-03,
    ##          7.73471766e-05,  5.76312467e-03],
    ##        [ 1.31810037e-02,  1.10727124e-05,  1.64337121e-02,
    ##          7.05591810e-04,  4.33882326e-03],
    ##        [ 1.43176480e-03,  1.93449866e-03,  1.49777310e-03,
    ##          9.49940551e-03,  2.43697385e-03],
    ##        [ 8.47901031e-03,  1.63551973e-04,  3.18611762e-03,
    ##         -2.28691783e-06,  1.14298956e-02],
    ##        [ 1.13125844e-02,  3.11139575e-03,  1.44482609e-02,
    ##          1.70786318e-03,  1.16892252e-02]], dtype=float32), 'biotic': array([0.9812051 , 0.9858    , 0.9686942 , 0.96637946, 0.992139  ,
    ##        0.9706108 , 0.9653298 , 0.9831996 , 0.97674376, 0.95773065],
    ##       dtype=float32)}
