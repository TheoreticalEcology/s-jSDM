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