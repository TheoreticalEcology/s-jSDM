## ---- echo = F, message = F----------------------------------------------
set.seed(123)

## ----global_options, include=FALSE---------------------------------------
knitr::opts_chunk$set(fig.width=7, fig.height=4.5, fig.align='center', warning=FALSE, message=FALSE, cache = F)

## ------------------------------------------------------------------------
library(sjSDM)

## ---- eval = F-----------------------------------------------------------
#  install_sjSDM()

## ------------------------------------------------------------------------
citation("sjSDM")

## ------------------------------------------------------------------------
com = simulate_SDM(env = 3L, species = 5L, sites = 100L)

## ------------------------------------------------------------------------
model = sjSDM(X = com$env_weights, Y = com$response, iter = 10L)

## ------------------------------------------------------------------------
coef(model)
summary(model)
getCov(model)

## ------------------------------------------------------------------------
model = sjSDM(X = com$env_weights, formula = ~ X1*X2,Y = com$response, iter = 10L)
summary(model)


## ------------------------------------------------------------------------
model = sjSDM(X = com$env_weights, formula = ~0+ I(X1^2),Y = com$response, iter = 10L)
summary(model)

