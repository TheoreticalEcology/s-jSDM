# Internal structure simulations

- [Internal community stucture
  simulations](#internal-community-stucture-simulations)
- [No effects, simple Sigma](#no-effects-simple-sigma)
- [With environmental effects and more covariance
  structure](#with-environmental-effects-and-more-covariance-structure)
- [With environmental and spatial effects and complex covariance
  structure](#with-environmental-and-spatial-effects-and-complex-covariance-structure)
  - [Compare spatial DNN with spatial
    LM](#compare-spatial-dnn-with-spatial-lm)

## Internal community stucture simulations

``` r
library(sjSDM)
```

## No effects, simple Sigma

``` r
set.seed(42)
SP = 4
Sigma = diag(1.0, SP)
diag(Sigma) = 1.0
Sigma[2,1] = Sigma[1,2] = 0.9
beta = c(rep(0, 4))
Env = rnorm(1000)
Y = 1*((Env %*% t(beta) + mvtnorm::rmvnorm(1000, sigma = Sigma))>0)
XY = matrix(rnorm(2000), 1000, 2)

fields::image.plot(Sigma)
```

![](Internal_files/figure-commonmark/unnamed-chunk-2-1.png)

``` r
model = sjSDM(Y = Y, 
              env = linear(matrix(Env, ncol = 1L)), 
              spatial = linear(XY, ~0+X1:X2),
              iter = 50L)
an = anova(model)
p = plot(an, internal = TRUE, suppress_plotting = TRUE)
p_shared = plot(an, internal = TRUE, add_shared = TRUE, suppress_plotting = TRUE)
```

# With environmental effects and more covariance structure

First five species only affected by environment, last 5 species with
weak environment and strong biotic components.

``` r
SP = 10
Sigma = diag(1.0, SP)
Sigma[] = 0.9
Sigma[1:5,] = Sigma[,1:5] = 0.0
diag(Sigma) = 1.0
fields::image.plot(Sigma)
```

![](Internal_files/figure-commonmark/unnamed-chunk-7-1.png)

``` r
beta = c(rep(1.5, 5), rep(0.3, 5))
Env = rnorm(500)
XY = matrix(rnorm(1000), 500, 2)
betaSP = rep(0.0, SP)
Y = 1*((Env %*% t(beta) + (XY[,1,drop=FALSE]*XY[,2,drop=FALSE]) %*% t(betaSP) +  mvtnorm::rmvnorm(500, sigma = Sigma))>0)
```

``` r
model = sjSDM(Y = Y, 
              env = linear(matrix(Env, ncol = 1L)), 
              spatial = linear(XY, ~0+X1:X2),
              iter = 100L)
an = anova(model)
p = plot(an, internal = TRUE, suppress_plotting = TRUE)
p_shared = plot(an, internal = TRUE, add_shared = TRUE, suppress_plotting = TRUE)
```

Separation is better with shared components?

# With environmental and spatial effects and complex covariance structure

First five species only affected by environment, last 5 species with
weak environment and strong biotic components. Species are alternately
affected by space:

``` r
SP = 10
Sigma = diag(1.0, SP)
Sigma[] = 0.9
Sigma[1:5,] = Sigma[,1:5] = 0.0
diag(Sigma) = 1.0
fields::image.plot(Sigma)
```

![](Internal_files/figure-commonmark/unnamed-chunk-12-1.png)

``` r
beta = c(rep(1.5, 5), rep(0.3, 5))
Env = rnorm(500)
XY = matrix(rnorm(1000), 500, 2)
betaSP = rep(c(0.1, 1.0), SP/2)
Y = 1*((Env %*% t(beta) + (XY[,1,drop=FALSE]*XY[,2,drop=FALSE]) %*% t(betaSP) +  mvtnorm::rmvnorm(500, sigma = Sigma))>0)
```

``` r
model = sjSDM(Y = Y, 
              env = linear(matrix(Env, ncol = 1L)), 
              spatial = linear(XY, ~0+X1:X2),
              iter = 100L)
an = anova(model)
p = plot(an, internal = TRUE, suppress_plotting = TRUE)
p_shared = plot(an, internal = TRUE, add_shared = TRUE, suppress_plotting = TRUE)
```

Again it seems that the shared components improve the separation!

## Compare spatial DNN with spatial LM

First five species only affected by environment, last 5 species with
weak environment and strong biotic components. Species are alternately
affected by space.

Space is modelled by DNN:

``` r
SP = 10
Sigma = diag(1.0, SP)
Sigma[] = 0.9
Sigma[1:5,] = Sigma[,1:5] = 0.0
diag(Sigma) = 1.0
fields::image.plot(Sigma)
```

![](Internal_files/figure-commonmark/unnamed-chunk-17-1.png)

``` r
beta = c(rep(1.5, 5), rep(0.3, 5))
Env = rnorm(500)
XY = matrix(rnorm(1000), 500, 2)
betaSP = rep(c(0.1, 1.0), SP/2)
Y = 1*((Env %*% t(beta) + (XY[,1,drop=FALSE]*XY[,2,drop=FALSE]) %*% t(betaSP) +  mvtnorm::rmvnorm(500, sigma = Sigma))>0)
```

``` r
model1 = sjSDM(Y = Y, 
              env = linear(matrix(Env, ncol = 1L)), 
              spatial = linear(XY, ~0+X1:X2),
              iter = 100L)
model2 = sjSDM(Y = Y, 
              env = linear(matrix(Env, ncol = 1L)), 
              spatial = DNN(XY, ~0+.),
              iter = 100L)
an1 = anova(model1)
an2 = anova(model2)
p_shared1 = plot(an1, internal = TRUE, add_shared = TRUE, suppress_plotting = TRUE)
p_shared2 = plot(an2, internal = TRUE, add_shared = TRUE, suppress_plotting = TRUE)
```

Almost identical! Which is good.
