
<!-- README.md is generated from README.Rmd. Please edit that file -->

[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![License: GPL
v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
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
devtools::install_github("https://github.com/TheoreticalEcology/s-jSDM", subdir = "sjSDM")
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

    ## LogLik:  -518.5017 
    ## Regularization loss:  0 
    ## 
    ## Species-species correlation matrix: 
    ## 
    ##  sp1  1.0000                                 
    ##  sp2 -0.3470  1.0000                             
    ##  sp3 -0.1460 -0.3670  1.0000                         
    ##  sp4 -0.1420 -0.3340  0.6970  1.0000                     
    ##  sp5  0.5890 -0.3290 -0.0930 -0.0910  1.0000                 
    ##  sp6 -0.2270  0.3710  0.1460  0.1490 -0.0520  1.0000             
    ##  sp7  0.4550 -0.1410  0.1280  0.1300  0.4940  0.2480  1.0000         
    ##  sp8  0.2790  0.1620 -0.4380 -0.4200  0.2300 -0.0310  0.1360  1.0000     
    ##  sp9 -0.0660 -0.0630  0.0730  0.0700 -0.3090 -0.2730 -0.2120 -0.1030  1.0000 
    ##  sp10     0.2000  0.3850 -0.6080 -0.5740  0.2340  0.1050  0.1230  0.4380 -0.2360  1.0000
    ## 
    ## 
    ## 
    ## Spatial: 
    ##             sp1        sp2       sp3       sp4       sp5       sp6       sp7
    ## X1:X2 0.2117761 -0.5411496 0.5273638 0.0178497 0.3714764 0.1739493 0.4265523
    ##             sp8       sp9      sp10
    ## X1:X2 0.4786434 0.2528224 0.2498071
    ## 
    ## 
    ## 
    ##                  Estimate  Std.Err Z value Pr(>|z|)    
    ## sp1 (Intercept)  -0.03613  0.20898   -0.17  0.86273    
    ## sp1 X1            0.85837  0.41098    2.09  0.03674 *  
    ## sp1 X2           -1.62117  0.38306   -4.23  2.3e-05 ***
    ## sp1 X3           -0.13945  0.34191   -0.41  0.68339    
    ## sp2 (Intercept)   0.00787  0.21249    0.04  0.97044    
    ## sp2 X1            0.89662  0.41437    2.16  0.03048 *  
    ## sp2 X2            0.16857  0.39930    0.42  0.67290    
    ## sp2 X3            0.44143  0.34029    1.30  0.19456    
    ## sp3 (Intercept)  -0.34590  0.21713   -1.59  0.11114    
    ## sp3 X1            0.99932  0.39444    2.53  0.01129 *  
    ## sp3 X2           -0.28363  0.38410   -0.74  0.46027    
    ## sp3 X3           -0.68942  0.36510   -1.89  0.05899 .  
    ## sp4 (Intercept)  -0.06449  0.20208   -0.32  0.74962    
    ## sp4 X1           -1.03488  0.38530   -2.69  0.00723 ** 
    ## sp4 X2           -1.31683  0.38969   -3.38  0.00073 ***
    ## sp4 X3           -0.29948  0.33239   -0.90  0.36760    
    ## sp5 (Intercept)  -0.12917  0.20334   -0.64  0.52526    
    ## sp5 X1            0.50595  0.38494    1.31  0.18872    
    ## sp5 X2            0.44272  0.37256    1.19  0.23470    
    ## sp5 X3           -0.44773  0.33606   -1.33  0.18276    
    ## sp6 (Intercept)   0.21687  0.20450    1.06  0.28891    
    ## sp6 X1            1.66739  0.42128    3.96  7.6e-05 ***
    ## sp6 X2           -0.70400  0.37734   -1.87  0.06209 .  
    ## sp6 X3            0.18265  0.33114    0.55  0.58124    
    ## sp7 (Intercept)   0.01107  0.20303    0.05  0.95651    
    ## sp7 X1           -0.27156  0.38754   -0.70  0.48348    
    ## sp7 X2            0.25865  0.37613    0.69  0.49165    
    ## sp7 X3           -1.06692  0.34221   -3.12  0.00182 ** 
    ## sp8 (Intercept)   0.15339  0.15690    0.98  0.32827    
    ## sp8 X1            0.27079  0.29912    0.91  0.36531    
    ## sp8 X2            0.30999  0.29261    1.06  0.28942    
    ## sp8 X3           -1.03282  0.27546   -3.75  0.00018 ***
    ## sp9 (Intercept)   0.01853  0.17201    0.11  0.91422    
    ## sp9 X1            1.09314  0.33410    3.27  0.00107 ** 
    ## sp9 X2           -0.80828  0.32387   -2.50  0.01257 *  
    ## sp9 X3            0.60859  0.28280    2.15  0.03140 *  
    ## sp10 (Intercept) -0.06874  0.18586   -0.37  0.71147    
    ## sp10 X1          -0.40988  0.33714   -1.22  0.22407    
    ## sp10 X2          -0.93544  0.34203   -2.73  0.00624 ** 
    ## sp10 X3          -0.38387  0.31449   -1.22  0.22223    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
plot(model)
```

    ## LogLik:  -518.5017 
    ## Regularization loss:  0 
    ## 
    ## Species-species correlation matrix: 
    ## 
    ##  sp1  1.0000                                 
    ##  sp2 -0.3470  1.0000                             
    ##  sp3 -0.1460 -0.3670  1.0000                         
    ##  sp4 -0.1420 -0.3340  0.6970  1.0000                     
    ##  sp5  0.5890 -0.3290 -0.0930 -0.0910  1.0000                 
    ##  sp6 -0.2270  0.3710  0.1460  0.1490 -0.0520  1.0000             
    ##  sp7  0.4550 -0.1410  0.1280  0.1300  0.4940  0.2480  1.0000         
    ##  sp8  0.2790  0.1620 -0.4380 -0.4200  0.2300 -0.0310  0.1360  1.0000     
    ##  sp9 -0.0660 -0.0630  0.0730  0.0700 -0.3090 -0.2730 -0.2120 -0.1030  1.0000 
    ##  sp10     0.2000  0.3850 -0.6080 -0.5740  0.2340  0.1050  0.1230  0.4380 -0.2360  1.0000
    ## 
    ## 
    ## 
    ## Spatial: 
    ##             sp1        sp2       sp3       sp4       sp5       sp6       sp7
    ## X1:X2 0.2117761 -0.5411496 0.5273638 0.0178497 0.3714764 0.1739493 0.4265523
    ##             sp8       sp9      sp10
    ## X1:X2 0.4786434 0.2528224 0.2498071
    ## 
    ## 
    ## 
    ##                  Estimate  Std.Err Z value Pr(>|z|)    
    ## sp1 (Intercept)  -0.03613  0.20898   -0.17  0.86273    
    ## sp1 X1            0.85837  0.41098    2.09  0.03674 *  
    ## sp1 X2           -1.62117  0.38306   -4.23  2.3e-05 ***
    ## sp1 X3           -0.13945  0.34191   -0.41  0.68339    
    ## sp2 (Intercept)   0.00787  0.21249    0.04  0.97044    
    ## sp2 X1            0.89662  0.41437    2.16  0.03048 *  
    ## sp2 X2            0.16857  0.39930    0.42  0.67290    
    ## sp2 X3            0.44143  0.34029    1.30  0.19456    
    ## sp3 (Intercept)  -0.34590  0.21713   -1.59  0.11114    
    ## sp3 X1            0.99932  0.39444    2.53  0.01129 *  
    ## sp3 X2           -0.28363  0.38410   -0.74  0.46027    
    ## sp3 X3           -0.68942  0.36510   -1.89  0.05899 .  
    ## sp4 (Intercept)  -0.06449  0.20208   -0.32  0.74962    
    ## sp4 X1           -1.03488  0.38530   -2.69  0.00723 ** 
    ## sp4 X2           -1.31683  0.38969   -3.38  0.00073 ***
    ## sp4 X3           -0.29948  0.33239   -0.90  0.36760    
    ## sp5 (Intercept)  -0.12917  0.20334   -0.64  0.52526    
    ## sp5 X1            0.50595  0.38494    1.31  0.18872    
    ## sp5 X2            0.44272  0.37256    1.19  0.23470    
    ## sp5 X3           -0.44773  0.33606   -1.33  0.18276    
    ## sp6 (Intercept)   0.21687  0.20450    1.06  0.28891    
    ## sp6 X1            1.66739  0.42128    3.96  7.6e-05 ***
    ## sp6 X2           -0.70400  0.37734   -1.87  0.06209 .  
    ## sp6 X3            0.18265  0.33114    0.55  0.58124    
    ## sp7 (Intercept)   0.01107  0.20303    0.05  0.95651    
    ## sp7 X1           -0.27156  0.38754   -0.70  0.48348    
    ## sp7 X2            0.25865  0.37613    0.69  0.49165    
    ## sp7 X3           -1.06692  0.34221   -3.12  0.00182 ** 
    ## sp8 (Intercept)   0.15339  0.15690    0.98  0.32827    
    ## sp8 X1            0.27079  0.29912    0.91  0.36531    
    ## sp8 X2            0.30999  0.29261    1.06  0.28942    
    ## sp8 X3           -1.03282  0.27546   -3.75  0.00018 ***
    ## sp9 (Intercept)   0.01853  0.17201    0.11  0.91422    
    ## sp9 X1            1.09314  0.33410    3.27  0.00107 ** 
    ## sp9 X2           -0.80828  0.32387   -2.50  0.01257 *  
    ## sp9 X3            0.60859  0.28280    2.15  0.03140 *  
    ## sp10 (Intercept) -0.06874  0.18586   -0.37  0.71147    
    ## sp10 X1          -0.40988  0.33714   -1.22  0.22407    
    ## sp10 X2          -0.93544  0.34203   -2.73  0.00624 ** 
    ## sp10 X3          -0.38387  0.31449   -1.22  0.22223    
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
    ## 1   1 0.8607188 3.239575e-04 0.13895726
    ## 2   2 0.5750836 3.900431e-03 0.42101595
    ## 3   3 0.7047927 3.665486e-03 0.29154167
    ## 4   4 0.9035402 2.383881e-06 0.09645750
    ## 5   5 0.5003806 2.760055e-03 0.49685937
    ## 6   6 0.8590981 2.233695e-04 0.14067854
    ## 7   7 0.5326309 1.682392e-03 0.46568665
    ## 8   8 0.8897696 3.936182e-03 0.10629424
    ## 9   9 0.7364383 5.591083e-04 0.26300257
    ## 10 10 0.9725977 1.205333e-03 0.02619702

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
    ## Biotic  156.64758         515.82178       0.79122      0.1133
    ## Abiotic 185.83240         329.98938       0.96744      0.2477
    ## Spatial   2.97705         327.01233       0.96840      0.2499

``` r
plot(an)
```

![](README_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

The anova shows the relative changes in the deviance of the groups and
their intersections.

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

    ## Iter: 0/10   0%|          | [00:00, ?it/s]Iter: 0/10   0%|          | [00:00, ?it/s, loss=7.321]Iter: 1/10  10%|#         | [00:00,  5.94it/s, loss=7.321]Iter: 1/10  10%|#         | [00:00,  5.94it/s, loss=7.321]Iter: 1/10  10%|#         | [00:00,  5.94it/s, loss=7.288]Iter: 1/10  10%|#         | [00:00,  5.94it/s, loss=7.288]Iter: 1/10  10%|#         | [00:00,  5.94it/s, loss=7.306]Iter: 1/10  10%|#         | [00:00,  5.94it/s, loss=7.283]Iter: 1/10  10%|#         | [00:00,  5.94it/s, loss=7.27] Iter: 1/10  10%|#         | [00:00,  5.94it/s, loss=7.289]Iter: 8/10  80%|########  | [00:00, 33.59it/s, loss=7.289]Iter: 8/10  80%|########  | [00:00, 33.59it/s, loss=7.255]Iter: 8/10  80%|########  | [00:00, 33.59it/s, loss=7.265]Iter: 10/10 100%|##########| [00:00, 32.14it/s, loss=7.265]

Calculate Importance:

``` python
Beta = np.transpose(model.env_weights[0])
Sigma = ( model.sigma @ model.sigma.t() + torch.diag(torch.ones([1])) ).data.cpu().numpy()
covX = fa.covariance( torch.tensor(Env).t() ).data.cpu().numpy()

fa.importance(beta=Beta, covX=covX, sigma=Sigma)
```

    ## {'env': array([[ 2.01624073e-03,  4.30547167e-03,  9.72515251e-03,
    ##          1.57106686e-02,  1.52279977e-02],
    ##        [ 7.17321597e-03,  9.13473498e-03,  3.63904328e-05,
    ##          8.77751689e-03,  5.95719670e-04],
    ##        [ 1.16564417e-02,  1.46691175e-02,  7.04750768e-04,
    ##          4.43916908e-03,  4.65077721e-03],
    ##        [ 9.54728015e-03,  1.32002533e-05,  7.40117254e-03,
    ##          1.19334860e-02,  1.51454210e-02],
    ##        [ 1.06326239e-02,  1.03140104e-04,  1.82005798e-03,
    ##          5.99095458e-03,  1.40121682e-02],
    ##        [-7.82942243e-06,  5.10592596e-04,  7.17978366e-03,
    ##          1.00550009e-02,  4.64413781e-03],
    ##        [ 5.81273716e-03,  8.43940210e-03, -3.09813622e-05,
    ##         -6.87289939e-05,  1.60868634e-02],
    ##        [ 1.39359059e-02,  1.84255268e-03,  8.07214249e-03,
    ##          1.18205510e-02,  1.13327922e-02],
    ##        [ 6.89498289e-03,  6.32241648e-03,  2.94192974e-03,
    ##          7.44746649e-05, -1.94817712e-05],
    ##        [ 1.35545945e-02,  4.42006765e-03,  5.76614868e-04,
    ##          8.61835666e-03,  8.24986864e-03]], dtype=float32), 'biotic': array([0.9530145 , 0.9742824 , 0.96387976, 0.9559594 , 0.9674411 ,
    ##        0.97761834, 0.9697607 , 0.9529961 , 0.9837856 , 0.9645805 ],
    ##       dtype=float32)}
