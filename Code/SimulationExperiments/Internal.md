# Internal structure simulations

- [<span class="toc-section-number">0.1</span> Internal community
  stucture simulations](#internal-community-stucture-simulations)
- [<span class="toc-section-number">0.2</span> No effects, simple
  Sigma](#no-effects-simple-sigma)
- [<span class="toc-section-number">1</span> With environmental effects
  and more covariance
  structure](#with-environmental-effects-and-more-covariance-structure)
- [<span class="toc-section-number">2</span> With environmental and
  spatial effects and complex covariance
  structure](#with-environmental-and-spatial-effects-and-complex-covariance-structure)
  - [<span class="toc-section-number">2.1</span> Compare spatial DNN
    with spatial LM](#compare-spatial-dnn-with-spatial-lm)

## Internal community stucture simulations

``` r
library(sjSDM)
set.seed(42)
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

<img src="Internal_files/figure-commonmark/fig-Fig_1-1.png"
id="fig-Fig_1"
alt="Figure  1: Internal structure. Left figure without shared components Right figure with shared contribution" />

<div id="tbl-Table_1">

| env | spa |    codist |        r2 |
|----:|----:|----------:|----------:|
|   0 |   0 | 0.1767892 | 0.0452041 |
|   0 |   0 | 0.1761780 | 0.0442250 |
|   0 |   0 | 0.0000000 | 0.0000000 |
|   0 |   0 | 0.0000000 | 0.0000000 |

**Table ** 1: Without shared components

</div>

<div id="tbl-Table_2">

|       env |       spa |    codist |        r2 |
|----------:|----------:|----------:|----------:|
| 0.0017425 | 0.0021448 | 0.1769292 | 0.0452041 |
| 0.0000000 | 0.0008602 | 0.1762018 | 0.0442250 |
| 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |

**Table ** 2: With shared components

</div>

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

<img src="Internal_files/figure-commonmark/fig-Fig_2-1.png"
id="fig-Fig_2"
alt="Figure  2: Internal structure. Left figure without shared components Right figure with shared contribution" />

<div id="tbl-Table_3">

|       env | spa |    codist |        r2 |
|----------:|----:|----------:|----------:|
| 0.4659596 |   0 | 0.1866608 | 0.0340383 |
| 0.3881666 |   0 | 0.1600174 | 0.0293152 |
| 0.4926410 |   0 | 0.2173934 | 0.0364265 |
| 0.4436425 |   0 | 0.1746357 | 0.0333473 |
| 0.4629698 |   0 | 0.1804307 | 0.0344467 |
| 0.0000000 |   0 | 0.4258151 | 0.0518634 |
| 0.0000000 |   0 | 0.3746577 | 0.0449380 |
| 0.0000000 |   0 | 0.4121028 | 0.0492912 |
| 0.0000000 |   0 | 0.4549209 | 0.0532570 |
| 0.0000000 |   0 | 0.3832169 | 0.0459453 |

**Table ** 3: Without shared components

</div>

<div id="tbl-Table_4">

|       env |       spa |    codist |        r2 |
|----------:|----------:|----------:|----------:|
| 0.3317235 | 0.0000000 | 0.0120565 | 0.0340383 |
| 0.2776112 | 0.0000000 | 0.0205169 | 0.0293152 |
| 0.3150089 | 0.0097533 | 0.0395031 | 0.0364265 |
| 0.3223509 | 0.0059607 | 0.0051611 | 0.0333473 |
| 0.3339948 | 0.0000000 | 0.0132398 | 0.0344467 |
| 0.0730981 | 0.0022953 | 0.4432406 | 0.0518634 |
| 0.0601361 | 0.0007967 | 0.3884474 | 0.0449380 |
| 0.0575959 | 0.0053011 | 0.4300150 | 0.0492912 |
| 0.0697708 | 0.0000000 | 0.4706310 | 0.0532570 |
| 0.0538345 | 0.0087622 | 0.3968565 | 0.0459453 |

**Table ** 4: With shared components

</div>

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

<img src="Internal_files/figure-commonmark/fig-Fig_3-1.png"
id="fig-Fig_3"
alt="Figure  3: Internal structure. Left figure without shared components Right figure with shared contribution" />

<div id="tbl-Table_5">

|       env |       spa |    codist |        r2 |
|----------:|----------:|----------:|----------:|
| 0.5090421 | 0.0000000 | 0.2169217 | 0.0392064 |
| 0.4014242 | 0.1626465 | 0.2231492 | 0.0436976 |
| 0.4562192 | 0.0000000 | 0.1957675 | 0.0324825 |
| 0.3081293 | 0.2068891 | 0.1805781 | 0.0387535 |
| 0.5492819 | 0.0000000 | 0.1978857 | 0.0405299 |
| 0.0000000 | 0.3757175 | 0.3749924 | 0.0519319 |
| 0.0000000 | 0.0000000 | 0.2983707 | 0.0415825 |
| 0.0000000 | 0.4542907 | 0.3819888 | 0.0548712 |
| 0.0000000 | 0.0000000 | 0.3033430 | 0.0448199 |
| 0.0000000 | 0.3851853 | 0.4014531 | 0.0551126 |

**Table ** 5: Without shared components

</div>

<div id="tbl-Table_6">

|       env |       spa |    codist |        r2 |
|----------:|----------:|----------:|----------:|
| 0.3996928 | 0.0000000 | 0.0026015 | 0.0392064 |
| 0.2780664 | 0.1238734 | 0.0350364 | 0.0436976 |
| 0.3248336 | 0.0000000 | 0.0174242 | 0.0324825 |
| 0.2333742 | 0.1473886 | 0.0067722 | 0.0387535 |
| 0.4166412 | 0.0040901 | 0.0000000 | 0.0405299 |
| 0.0548115 | 0.1015240 | 0.3629837 | 0.0519319 |
| 0.0492140 | 0.0359590 | 0.3306524 | 0.0415825 |
| 0.0541316 | 0.1409709 | 0.3536091 | 0.0548712 |
| 0.0550197 | 0.0502657 | 0.3429132 | 0.0448199 |
| 0.0545512 | 0.1111569 | 0.3854178 | 0.0551126 |

**Table ** 6: With shared components

</div>

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

<img src="Internal_files/figure-commonmark/fig-Fig_4-1.png"
id="fig-Fig_4"
alt="Figure  4: Internal structure. Left figure with spatial LM. Right figure with spatial DNN" />

<div id="tbl-Table_7">

|       env |       spa |    codist |        r2 |
|----------:|----------:|----------:|----------:|
| 0.4633396 | 0.0481404 | 0.0000000 | 0.0414589 |
| 0.3545260 | 0.0825792 | 0.0023565 | 0.0439462 |
| 0.4166600 | 0.0651157 | 0.0000000 | 0.0401109 |
| 0.2946496 | 0.0589566 | 0.0426710 | 0.0396277 |
| 0.4252812 | 0.0602266 | 0.0000000 | 0.0404997 |
| 0.0801862 | 0.0000000 | 0.5288423 | 0.0539402 |
| 0.0514171 | 0.1180636 | 0.3609658 | 0.0530447 |
| 0.0750475 | 0.0000000 | 0.5698943 | 0.0595402 |
| 0.0563420 | 0.1296816 | 0.3372693 | 0.0523293 |
| 0.0843799 | 0.0000000 | 0.5600249 | 0.0569835 |

**Table ** 7: Spatial LM (with shared components)

</div>

<div id="tbl-Table_8">

|       env |       spa |    codist |        r2 |
|----------:|----------:|----------:|----------:|
| 0.4827951 | 0.0000000 | 0.0000000 | 0.0433513 |
| 0.2829020 | 0.0643807 | 0.0514862 | 0.0398769 |
| 0.4132390 | 0.0010131 | 0.0000000 | 0.0398327 |
| 0.2021344 | 0.0941536 | 0.0842820 | 0.0380570 |
| 0.4383528 | 0.0000000 | 0.0000000 | 0.0418536 |
| 0.0427857 | 0.1232256 | 0.3746952 | 0.0540706 |
| 0.0573697 | 0.0559144 | 0.4350481 | 0.0548332 |
| 0.0449463 | 0.1112383 | 0.4417945 | 0.0597979 |
| 0.0596575 | 0.0553295 | 0.4070881 | 0.0522075 |
| 0.0403012 | 0.1213577 | 0.4045271 | 0.0566186 |

**Table ** 8: Spatial DNN (with shared components)

</div>

Almost identical! Which is good.
