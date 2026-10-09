# sjSDM 1.1.0

## Major changes

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
* `RMSprop()` and `SGD()` now use 'torch''s `optim_ignite_rmsprop` and `optim_ignite_sgd`,
  which run in libtorch rather than in R. They are bit-identical to the implementations they
  replace (verified per optimizer step across momentum, centered, nesterov and dampening) and
  2-3.5x faster per step. The other five optimizers have no counterpart in 'torch' and are
  unchanged.
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