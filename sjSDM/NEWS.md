# sjSDM 1.1.0

## Major changes

* `NN()` is new. It marks a group of covariates inside an environmental formula as a
  non-linear block, so that a flexible term can sit *beside* interpretable linear terms
  rather than replacing them: `linear(X, ~ temp + NN(soil + ndvi, hidden = c(50, 50)))`.
  The blocks are summed on the linear-predictor scale, so `coef()`, `summary()` and the
  standard errors still describe the parametric part; the standard errors are conditional
  on the fitted block and a warning says so. A covariate may not appear both in the
  parametric part and in an `NN()` block, `NN()` may not appear inside an interaction, and
  it is not yet supported in the spatial formula. `update()` gains `env_blocks` to refit
  with a subset of the blocks, which is how a block is switched off.
* Grouping-factor random effects are new. `lme4` bars are written directly in the
  environmental formula and carry `lme4`'s own semantics, through the 'reformulas' package:
  `linear(X, ~ temp + (1 | plot))`, `(temp | plot)` for a correlated intercept and slope,
  `(0 + temp | plot)`, `(temp || plot)` and `(1 | plot/subplot)`. `re()` is the explicit
  form where a bar needs arguments, `re(1 | plot, df = 2)`.
* A bar is a design-level latent factor, not a scalar shift: the group effect carries its
  own species-species covariance `Lambda Lambda'` of rank `df`, so species may respond to a
  plot with different magnitude and sign. `df` defaults to `max(q, 5)`, where `q` is the
  number of columns of the bar, and must not be smaller than `q` — at `df < q` the
  intercept-slope correlation would be forced to exactly +-1. `loading = "shared"` gives
  the classical mixed-model case instead, one shift per group identical for every species.
* The bars are integrated out by a stochastic variational approximation: each group gets a
  Gaussian posterior drawn with the reparameterisation trick, and the Kullback-Leibler term
  to the `N(0, I)` prior supplies the shrinkage. The Monte-Carlo likelihood itself is
  unchanged. The variational parameters are optimised in their own parameter group with
  `weight_decay = 0`; with the shared decay, groups absent from a minibatch were shrunk to
  zero by the optimiser rather than by the data.
* `getRE()` returns, per bar, the group covariance, the predicted group effects, the
  loadings and the posterior Cholesky factor. `getCov()` and `getCor()` gain
  `which = c("residual", "re")`; `"re"` returns one entry per bar, named by the grouping
  variable.
* Known limitations of the random effects. `anova()` refuses on a model containing a bar:
  the block is re-estimated in every refit and the per-species rescaling is recomputed with
  it, so the fractions would not decompose. `logLik()` for such a model is the conditional
  value at the posterior means, so it is not comparable with a model fitted without a bar.
  `sjSDM_cv()` does not support bars or `NN()` blocks and raises an error rather than
  dropping them. Bars are not available in the spatial formula. The reported group variance
  grows with `df`, because the Kullback-Leibler term constrains the posterior and not the
  loadings; report the rank alongside the variance, and prefer a small `df`.
* The backend was rewritten. sjSDM now runs on the R 'torch' package (libtorch) and no
  longer depends on 'python', 'conda', 'reticulate' or 'PyTorch'. `install_sjSDM()` is
  now a thin wrapper around `torch::install_torch()`; the 'r-sjsdm' conda environment is
  no longer used and can be deleted.
* Results are not bit-identical to the 'PyTorch' backend. The two backends use different
  random number streams, so weight initialisation and the Monte-Carlo draws differ.
  Coefficients, associations and log-likelihoods agree within Monte-Carlo noise
  (validated against 1.0.7 across all families, spatial and DNN models).
* `install_sjSDM()` no longer takes a `version` argument; whether the GPU is used is
  decided by the 'torch' installation. `device = "mps"` now works on Apple silicon.
* `sjSDMControl(mixed = TRUE)` (half precision) is accepted but ignored.
* `model$history` is a `data.frame` with columns `epoch` and `train_l`, not a numeric
  vector, and it holds only the epochs that ran. An early-stopped fit used to leave the
  remaining epochs as zeros, which `plot()` drew as part of the loss curve. Code that
  indexed `model$history` as a vector has to read `model$history$train_l`.
* `Adam()`, `AdamW()` and `Adagrad()` are new and are the recommended optimizers.
  `Adamax()`, `AdaBound()`, `DiffGrad()` and `madgrad()` have no counterpart in 'torch',
  redirect to `Adam()` with a message, and `AccSGD()` redirects to `SGD()`. They are kept
  only because they are part of the 1.0.7 API and will be removed in 1.2.0.
* A GPU request is now validated before anything is allocated. `device = 0L`, `"gpu"`,
  `"cuda"` and `"cuda:0"` all fall back to the CPU with a warning when no CUDA device is
  present, and an index larger than the number of installed devices is an error instead of
  a backend failure inside libtorch. The check sits in the one place every model rebuild
  goes through, so it also covers `anova()`, `sjSDM_cv()` and a reloaded model.

## Bug fixes

* `getSe()` did not pass the spatial predictors to the Hessian, so standard errors for
  spatial models were computed as if the spatial term were absent. `plot()` on a spatial
  model silently fell back to no standard errors as a result.
* An integer response matrix containing `NA` was converted to the integer minimum instead
  of `NaN`, which silently corrupted the missing-value masking and conditional predictions.
  Responses are now always converted through double.
* `setWeights(model)` with the default `weights = NULL` errored with a subscript out of
  bounds for non-spatial models.
* `se()` dropped the last incomplete batch, so the Hessian was accumulated over fewer
  observations than the model was fitted on.
* `predict(type = "raw")` returned `0.999999 * mu + 5e-7` instead of `mu`; the
  probability guard is now only applied to bounded links.
* The probability guard was also applied to Poisson, negative binomial and Gaussian
  rates in `MVP_logLik`, which is not meaningful for an unbounded mean.
* The Hessian ridge in `se()` was added to every element of the matrix rather than to its
  diagonal.
* A `verbose` argument passed to `sjSDM_cv()` errored with "matched by multiple actual
  arguments".
* Fitting with `step_size` larger than the number of observations silently trained on no
  batches at all.
* **Marginal predictions were not marginal.** `predict()` applied the link to the linear
  predictor directly, which is the prediction at latent `z = 0`, not the expectation over
  the latent factor that generates the species associations. The marginal is
  `E_z[link(mu + z sigma')]`: for probit `pnorm(mu / sqrt(diag(sigma sigma' + I)))`, for the
  log links `exp(mu + rowSums(sigma^2)/2)`, and unchanged for gaussian. The error grew with
  the strength of the associations — up to 0.12 in probability for probit, and a factor of
  roughly 1.5 on the rate for poisson and nbinom. `predict()` now returns the marginal;
  `predict(marginal = FALSE)` restores the old behaviour for reproducing earlier results.
  probit, poisson, nbinom and gaussian use exact closed forms and stay deterministic; logit
  and linear have no closed form and are approximated by Monte Carlo, so they are stochastic
  and respond to `sampling`.
* **R squared values change as a result.** `anova()` measures against a null model fitted
  with `bioticStruct(diag = TRUE)`, which is `sigma = I` and therefore `diag(getCov) = 2`,
  and it takes `predict(null_model)` as the null probability. With the corrected marginal the
  null likelihood shifts, moving McFadden and Nagelkerke R squared by roughly a quarter of
  their value in tests. Variation-partitioning results from earlier versions are not
  comparable.
* Conditional predictions (`predict(model, Y = ...)`) failed for a model with two species
  when one of them was conditioned on, with `IndexError: tuple index out of range`. With a
  single conditioning species `Y[, focal]`, `predictions[, focal]` and `sigma[focal, ]` all
  dropped to vectors, so the backend was handed a one-dimensional association matrix. Three
  or more species hid it, because the conditioning set was then never a single column. The
  torch backend keeps the matrix shape and the case is covered by a regression test.
* **`anova()` did not switch a DNN component off.** The `~0` refits build a design matrix
  with zero columns, but only the first layer of a deep component was built without a bias,
  so a switched-off DNN still emitted a learned species constant out of its output layer and
  the fraction attributed to it was not zero. Linear components were never affected. On a
  model with a DNN environmental and a DNN spatial component, McFadden `F_S` moves
  0.01422 -> 0.00438, `F_AS` 0.00027 -> 0.02454, `F_ABS` -0.01161 -> -0.03448, the saturated
  R squared 0.9232 -> 0.8296 and the null log-likelihood -488.470 -> -488.489. Variation
  partitioning of DNN models from earlier versions is not comparable.
* **`predict(newdata = )` re-derived a data-dependent basis from the new rows.** The fitted
  model now carries its `terms` object and factor levels, so `poly()`, `ns()`, `scale()` and
  factor contrasts are evaluated exactly as they were at fit time. Predictions from a model
  whose formula contains any of these were wrong on new data and silently so; predictions on
  the training data were unaffected.
* **`sjSDM_cv()` failed for any formula with an interaction.** Each fold overwrites the
  design matrix with a sliced, already expanded one and sets `formula = ~0+.` without
  rebuilding the stored terms, so `predict(newdata = )` evaluated the original formula
  against the expanded design and failed with "object 'X1' not found" on, for example,
  `~0+X1:X2`. The default formula passed only by coincidence.
* **`sjSDM_cv()` penalised the intercept in every fold.** Whether the intercept is excluded
  from the L1/L2 penalty was derived from the string `"(Intercept)"` in the design's column
  names, and the per-fold design is rebuilt through `data.frame()`, which renames that
  column. With `lambda_coef = 0.5` on a 60 x 5 problem the fold log-likelihood moved
  208.3551 -> 211.1313 and `AUC_test` 0.4330 -> 0.4082. The flag is now carried on the
  `linear()` / `DNN()` object instead of being re-derived from a column name, and the
  `lambda = 0` arm was never affected.
* `setWeights()` was undone by the next `saveRDS()` / `readRDS()`. It changed the live
  'torch' module, but a saved model is rebuilt from the serialised state captured at fit
  time, so the injected weights were silently replaced by the fitted ones on reload
  (measured maximum deviation 1.089 in the prediction). `setWeights()` now refreshes that
  state as well.
* `getSe()` now fails with a clear message on a model with a `DNN()` environmental
  component. Standard errors are only defined for the linear model; the Hessian code
  previously errored inside 'torch' when the number of species exceeded the width of the
  first hidden layer, and was silently wrong otherwise.
* `anova()`'s leave-one-out likelihood used a single species' row of the association matrix,
  broadcast over the remaining species, instead of every species but one. Introduced during
  the torch port: 'torch' follows python's negative-index semantics, so `sigma[-i, ]` is the
  i-th row counted from the end rather than "all rows except i". The aggregate variation
  partitioning was almost unaffected, but the per-species decomposition that
  `internalStructure()` builds on was systematically wrong.

## Minor changes

* The Monte-Carlo reduction in the likelihood uses `torch_logsumexp()` instead of a
  hand-rolled max-shift, which replaces seven kernel launches with three. `torch_logsumexp`
  performs the same max-subtraction internally, so the computation is unchanged; the
  fixed-noise parity against the 'PyTorch' backend agrees to 1e-8 for all six links.
* Every optimizer is now one of 'torch''s own `optim_ignite_*` builds, which run in libtorch
  rather than in R. `RMSprop()` and `SGD()` are bit-identical to the hand-written
  implementations they replace (verified per optimizer step across momentum, centered,
  nesterov and dampening) and 2-3.5x faster per step; the five with no 'torch' counterpart
  redirect, see Major changes.
* 'iml' was dropped from `Suggests`. It was a leftover from the removed `importance()` and
  was used nowhere in the package, but `R CMD check` refuses to run without an installed
  suggested package.


# sjSDM 1.0.7

* importance function was removed (deprecated)

## Bug fixes
* cv function, use correct indices in parallel block modus #155, thanks to @jannebor
* remove ggtern dependency


# sjSDM 1.0.6
## New features
* Datasets: butterflies and eucalypt species
* Conditional predictions
* Assembly regression plots


# sjSDM 1.0.5
## New features
* Pass custom test indices to sjSDM function via CV argument
* improve reproducibility (seeding)
* improve stability of ANOVA

## Bug fixes
* fixed small bug in the calculation of the partial Rsquareds

# sjSDM 1.0.4
## New features

* Anova function is now based on conditional probabilities to better separate 
  the biotic components
* Anova can use shared components (plot(...,internal=TRUE, add_shared=TRUE))
* Simulation methods for all supported families (binomial, poisson, and
  negative binomial)
* Support for negative binomial distribution
* plotInternalStructure for plotting internal metacommunity structure
* getCor to return species-species association matrix


## Bug fixes

* fixed Rsquared(...) #113 (thanks to @AndrewCSlater)
* fixed whitespaces in species names #115 @dansmi-hub

# sjSDM 1.0.3
## Minor changes

* changed weight_decay in 'RMSprop' from 0.01 to 0.0001 

## Important bug fix

* fixed sjSDM_cv(...) #104 (thanks to @Cdevenish)



# sjSDM 1.0.2
## Minor changes
* changed \mjeqn{}{} to \mjeqn{} as requested by the CRAN team

## Bug fixes

* fixed plot.sjSDM(...) #97



# sjSDM 1.0.1

## New Features

* anova plots for internal meta-community structure (based on individual R-squared values)

## Minor changes

* first layer of DNN now always without an explicit bias (bias/intercept is passed by model/formula, if desired)
* revised prediction function, improved stability
* revised simulation function, samples now from a multivariate probit model

## Bug fixes

* unlisting of config objects in `sjSDM::sjSDM_cv` (thanks to Máté) (added unit tests)  #88
* `sjSDM::Rsquared` bug for spatial models (thanks to Máté) #90
* revised regularization behavior, l1 and l2 were not correctly imposed on DNN structure
* revised and improved setWeights function
* bugs in vignettes (thanks to Doug) #92
* bugs in plot function for models with DNN objects




# sjSDM 1.0.0

## Major changes

* revised anova: `sjSDM::anova(...)` corresponds now to a type I anova (removed CV) #76
* `sjSDM::Rsquared()` uses now Nagelkerke or McFadden R-squared (which is also used in the anova) #76
* deprecated `sjSDM::sLVM` because of instability issues and other reasons
* revised `sjSDM::install_sjSDM()`, it works now for all x64 systems/versions #81 #79 #71

## Minor changes

* removed several unnecessary dependencies (e.g. dplyr)
* improved documentation of all functions, e.g. see `?sjSDM`
* new `sjSDM::update.sjSDM` method to re-fit model with different formula(s)
* new `sjSDM::sjSDM.tune` method to fit quickly a model with optimized regularization parameters (from `sjSDM::sjSDM_cv`)

## Bug fixes

* revised memory problem in `sjSDM::sjSDM_cv()` #84