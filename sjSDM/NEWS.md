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