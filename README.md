---
output: github_document
---

<!-- README.md is generated from README.Rmd. Please edit that file -->


[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/sjSDM)](https://cran.r-project.org/package=sjSDM) ![R-CMD-check](https://github.com/TheoreticalEcology/s-jSDM/workflows/R-CMD-check/badge.svg?branch=master) [![Publication](https://img.shields.io/badge/Publication-10.1111/2041-green.svg)](https://www.doi.org/10.1111/2041-210X.13687)

# s-jSDM - Fast and accurate Joint Species Distribution Modeling

## About sjSDM

The sjSDM package is an R package for estimating joint species distribution models. A jSDM is a GLMM that models a multivariate (i.e. a many-species) response to the environment, space and a covariance term that models conditional (on the other terms) correlations between the outputs (i.e. species). 

![image](sjSDM/vignettes/jSDM-structure.png)

A big challenge in jSDM implementation is computational speed. The goal of the sjSDM (which stands for "scalable joint species distribution models") is to make jSDM computations fast and scalable. Unlike many other packages, which use a latent-variable approximation to make estimating jSDMs faster, sjSDM fits a full covariance matrix in the likelihood, which is, however, numerically approximated via simulations. The method is described in Pichler & Hartig (2021) A new joint species distribution model for faster and more accurate inference of species associations from big community data, https://www.doi.org/10.1111/2041-210X.13687. 

Since version 1.1.0 the core code of sjSDM is implemented in R on top of [torch](https://torch.mlverse.org/) (libtorch). Earlier versions wrapped a Python / PyTorch backend through reticulate; that is gone, and with it the need for python, conda and a separate PyTorch installation. Model fits are unchanged apart from Monte-Carlo noise. `anova()` is an exception: 1.1.0 fixes several defects in it that move variation-partitioning results, see `news(package = "sjSDM")`.

To get citation info for sjSDM when you use it for your reseach, type  


``` r
citation("sjSDM")
```

## Installing the R package

sjSDM is distributed via [CRAN](https://cran.rstudio.com/web/packages/sjSDM/index.html). For most users, it will be best to install the package from CRAN


``` r
install.packages("sjSDM")
```

The only dependency that is not an ordinary R package is the libtorch binary, which `torch` downloads on first use. If that has not happened yet, run


``` r
sjSDM::install_sjSDM()   # wrapper around torch::install_torch()
```

There is no separate CPU and GPU build any more: whether the GPU can be used follows from the `torch` installation (`torch::cuda_is_available()`, or `torch::backends_mps_is_available()` on Apple silicon). Pass `device = "gpu"` or `device = "mps"` to `sjSDM()` to use it.

For advanced users: if you want to install the current (development) version from this repository, run


``` r
devtools::install_github("https://github.com/TheoreticalEcology/s-jSDM", subdir = "sjSDM", ref = "master")
```

If the installation fails, check out the help of ?install_sjSDM and ?installation_help.

1.  Run `sjSDM::install_diagnostic()` and check whether libtorch is installed
2.  If not, run `torch::install_torch()` and restart R
3.  If you still do not get the pkg to run, create an issue on the [issue tracker](https://github.com/TheoreticalEcology/s-jSDM/issues) or write an email to maximilian.pichler at ur.de

## Basic Workflow

Load the package


``` r
library(sjSDM)
```

Simulate some community data 


``` r
set.seed(42)
community <- simulate_SDM(sites = 100, species = 10, env = 3, se = TRUE)
Env <- community$env_weights
Occ <- community$response
SP <- matrix(rnorm(200, 0, 0.3), 100, 2) # spatial coordinates (no effect on species occurences)
```

This fits the standard SDM with environmental, spatial and covariance terms 


``` r
model <- sjSDM(Y = Occ, env = linear(data = Env, formula = ~X1+X2+X3), spatial = linear(data = SP, formula = ~0+X1:X2), se = TRUE, family=binomial("probit"), sampling = 100L, verbose = FALSE)
```


``` r
summary(model)
```

```
## Family:  binomial 
## 
## LogLik:  -503.8592 
## Regularization loss:  0 
## 
## Species-species correlation matrix: 
## 
## 	sp1	 1.0000									
## 	sp2	-0.3810	 1.0000								
## 	sp3	-0.1950	-0.4020	 1.0000							
## 	sp4	-0.1810	-0.3680	 0.8110	 1.0000						
## 	sp5	 0.6740	-0.3740	-0.1320	-0.1130	 1.0000					
## 	sp6	-0.3110	 0.4810	 0.1760	 0.1940	-0.0970	 1.0000				
## 	sp7	 0.5490	-0.1050	 0.1130	 0.1420	 0.5440	 0.2650	 1.0000			
## 	sp8	 0.2570	 0.1990	-0.5030	-0.5010	 0.2170	-0.0670	 0.0870	 1.0000		
## 	sp9	-0.0760	-0.0500	 0.0580	 0.0420	-0.4100	-0.3410	-0.2510	-0.1010	 1.0000	
## 	sp10	 0.2250	 0.4580	-0.6980	-0.6540	 0.2740	 0.1340	 0.1610	 0.4660	-0.2760	 1.0000
## 
## 
## 
## Spatial: 
##            sp1       sp2      sp3       sp4      sp5       sp6     sp7      sp8
## X1:X2 1.645019 -3.182818 2.972885 0.5640423 2.251627 0.9544154 2.98084 1.918683
##            sp9     sp10
## X1:X2 1.020852 1.026658
## 
## 
## 
##                  Estimate  Std.Err Z value Pr(>|z|)    
## sp1 (Intercept)  -0.04997  0.25093   -0.20  0.84216    
## sp1 X1            1.26035  0.48592    2.59  0.00949 ** 
## sp1 X2           -2.29708  0.44545   -5.16  2.5e-07 ***
## sp1 X3           -0.16742  0.41938   -0.40  0.68974    
## sp2 (Intercept)  -0.00757  0.26008   -0.03  0.97677    
## sp2 X1            1.22945  0.51076    2.41  0.01608 *  
## sp2 X2            0.26143  0.47187    0.55  0.57956    
## sp2 X3            0.59345  0.43289    1.37  0.17040    
## sp3 (Intercept)  -0.46861  0.26464   -1.77  0.07660 .  
## sp3 X1            1.32720  0.48628    2.73  0.00635 ** 
## sp3 X2           -0.43543  0.48236   -0.90  0.36667    
## sp3 X3           -0.99407  0.47882   -2.08  0.03789 *  
## sp4 (Intercept)  -0.04630  0.23260   -0.20  0.84221    
## sp4 X1           -1.44001  0.45007   -3.20  0.00138 ** 
## sp4 X2           -1.79183  0.46679   -3.84  0.00012 ***
## sp4 X3           -0.36523  0.40532   -0.90  0.36754    
## sp5 (Intercept)  -0.20750  0.26492   -0.78  0.43348    
## sp5 X1            0.71878  0.50462    1.42  0.15434    
## sp5 X2            0.55711  0.49486    1.13  0.26026    
## sp5 X3           -0.66421  0.42339   -1.57  0.11670    
## sp6 (Intercept)   0.26138  0.24523    1.07  0.28648    
## sp6 X1            2.32184  0.49322    4.71  2.5e-06 ***
## sp6 X2           -0.97278  0.45923   -2.12  0.03415 *  
## sp6 X3            0.11280  0.40215    0.28  0.77909    
## sp7 (Intercept)  -0.01080  0.25176   -0.04  0.96578    
## sp7 X1           -0.32307  0.46673   -0.69  0.48881    
## sp7 X2            0.34819  0.45655    0.76  0.44566    
## sp7 X3           -1.42476  0.42948   -3.32  0.00091 ***
## sp8 (Intercept)   0.13233  0.15820    0.84  0.40289    
## sp8 X1            0.32278  0.30744    1.05  0.29377    
## sp8 X2            0.30161  0.29871    1.01  0.31263    
## sp8 X3           -1.15826  0.28079   -4.13  3.7e-05 ***
## sp9 (Intercept)   0.03055  0.19389    0.16  0.87481    
## sp9 X1            1.34476  0.38191    3.52  0.00043 ***
## sp9 X2           -1.02651  0.37237   -2.76  0.00584 ** 
## sp9 X3            0.77492  0.32625    2.38  0.01754 *  
## sp10 (Intercept) -0.10586  0.19967   -0.53  0.59599    
## sp10 X1          -0.51578  0.36281   -1.42  0.15514    
## sp10 X2          -1.21615  0.36411   -3.34  0.00084 ***
## sp10 X3          -0.54970  0.33679   -1.63  0.10264    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```


Plot the niche estimates, i.e the estimates in the environmental component:


``` r
plot(model)
```

```
## Warning: Using `size` aesthetic for lines was deprecated in ggplot2 3.4.0.
## ℹ Please use `linewidth` instead.
## ℹ The deprecated feature was likely used in the sjSDM package.
##   Please report the issue at
##   <https://github.com/TheoreticalEcology/s-jSDM/issues>.
## This warning is displayed once per session.
## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
## generated.
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-1.png)

Visualize the species-species association matrix


``` r
image(getCor(model))
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10-1.png)


## Anova / Variation partitioning

### Global ANOVA

As in other models, it can be interesting to analyze how much variation is explained by which parts of hte model. 

![image](sjSDM/vignettes/jSDM-ANOVA.png){{width=70%}}
For the Env, Spatial, Covariance terms, this is implemented in 


``` r
an = anova(model, verbose = FALSE)
```



``` r
summary(an)
```

```
## Analysis of Deviance Table
## 
##                Deviance Residual deviance R2 Nagelkerke R2 McFadden
## Abiotic       155.85856        1226.60906       0.78957      0.1127
## Associations  202.98152        1179.48611       0.86864      0.1468
## Spatial        17.75269        1364.71494       0.16266      0.0128
## Full          376.59277        1005.87485       0.97685      0.2724
```

``` r
plot(an)
```

![plot of chunk unnamed-chunk-12](figure/unnamed-chunk-12-1.png)

The anova shows the relative changes in the R^2^ of the groups and their intersections.

### Internal metacommunity structure

Following [Leibold et al., 2022](https://doi.org/10.1111/oik.08618) we can calculate and visualize the internal metacommunity structure (=partitioning of the three components for species and sites). The internal structure is already calculated by the ANOVA and we can visualize it with the plot method:


``` r
results = internalStructure(an) # or plot(an, internal = TRUE)
```

The plot function returns the results for the internal metacommunity structure:


``` r
plot(results)
```

![plot of chunk unnamed-chunk-14](figure/unnamed-chunk-14-1.png)

Which can be regressed against covariates to analyse assembly processes:


``` r
plotAssemblyEffects(results)
```

![plot of chunk unnamed-chunk-15](figure/unnamed-chunk-15-1.png)


