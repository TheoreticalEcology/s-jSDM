## Version 1.0.4
### Submission 1, XX/04/2023


## Version 1.0.3
### Submission 1, 14/09/2022
This is a minor update, fixing a bug in the tuning function (sjSDM_cv) and
changing weight_decay in 'RMSprop' from 0.01 to 0.0001. Compatibility 
with PyTorch 1.12.1 was successfully checked.



### Successfull R CMD checks under
* Locally: MacOS Monterey 12.5.1 (R x86_64 version)
* Github actions: 
  - MacOS Catalina 10.15 R-release 
  - Ubuntu 20.04 R-release, R-oldrelease, and R-development
  - Windows-latest R-release
* Rhub:
  - fedora 
* Win-builder R-release, R-development, and R-oldrelease

Notes from win-builder:
Found the following (possibly) invalid URLs:
  URL: https://doi.org/10.1111/2041-210X.13687
    From: README.md
    Status: 503
    Message: Service Unavailable

Found the following (possibly) invalid DOIs:
  DOI: 10.1111/2041-210X.13687
    From: DESCRIPTION
          inst/CITATION
    Status: Service Unavailable
    Message: 503

URL and DOI are correct.

Possibly misspelled words in DESCRIPTION:
  CPUs (21:262)
  GPUs (21:271)
  Hartig (21:320)
  Pichler (21:310)
  Scalable (3:8)
  autocorrelation (21:488)
  jSDMs (21:79, 21:565)
  scalable (21:16)

The spelling is intended.





## Version 1.0.2
### Submission 1, 06/22/2022
This is a minor update, fixing a minor bug in the plot function and changing 
'\mjeqn{}{}' to '\mjeqn{}' as requested by the CRAN team

### Successfull R CMD checks under
* Locally: MacOS Monterey 12.4 (R x86_64 version)
* Github actions: 
  - MacOS Catalina 10.15 R-release 
  - Ubuntu 20.04 R-release, R-oldrelease, and R-development
  - Windows-latest R-release
* Rhub:
  - fedora 
  - solairs
* Win-builder R-release, R-development, and R-oldrelease

Notes from win-builder:
Found the following (possibly) invalid URLs:
  URL: https://doi.org/10.1111/2041-210X.13687
    From: README.md
    Status: 503
    Message: Service Unavailable

Found the following (possibly) invalid DOIs:
  DOI: 10.1111/2041-210X.13687
    From: DESCRIPTION
          inst/CITATION
    Status: Service Unavailable
    Message: 503

URL and DOI are correct.

Possibly misspelled words in DESCRIPTION:
  CPUs (21:262)
  GPUs (21:271)
  Hartig (21:320)
  Pichler (21:310)
  Scalable (3:8)
  autocorrelation (21:488)
  jSDMs (21:79, 21:565)
  scalable (21:16)

The spelling is intended.





## Version 1.0.1

### Submission 1, 03/11/2022
This is a minor update, fixing a few minor bugs and undesired behaviors reported 
by users and adding a few new functionalities to analyse the internal 
meta-community structure.

Examples cannot be executed directly after the pkg installation
because they need additional python dependencies which first have to be 
installed by the `sjSDM::install_sjSDM()` function. Therefore we deiced to use
the `\dontrun{}` flag


### Successfull R CMD checks under

* Locally: MacOS Monterey 12.2.1 (R x86_64 version)
* Github actions: 
  - MacOS Catalina 10.15 R-release 
  - Ubuntu 20.04 R-release, R-oldrelease, and R-development
  - Windows-latest R-release
* Rhub:
  - fedora 
  - solairs
* Win-builder R-release, R-development, and R-oldrelease

Notes (from win-builder): 

Possibly misspelled words in DESCRIPTION:
  CPUs (21:262)
  GPUs (21:271)
  Hartig (21:320)
  Pichler (21:310)
  Scalable (3:8)
  autocorrelation (21:488)
  jSDMs (21:79, 21:565)
  scalable (21:16)

The spelling is intended.




## Version 1.0.0


### Submission 2, 01/07/2022

Hi,

thanks for the comments! I have addressed your points below:

Best,
Max

> Please add \value to .Rd files regarding exported methods and explain
> the functions results in the documentation. Please write about the
> structure of the output (class) and also what the output means. (If a
> function does not return a value, please document that too, e.g.
> \value{No return value, called for side effects} or similar)
> Missing Rd-tags in up to 41 .Rd files, e.g.:
> AccSGD.Rd: \value
> AdaBound.Rd: \value
> Adamax.Rd: \value
> add_legend.Rd: \value
> add_species_arrows.Rd: \value
> bioticStruct.Rd: \value
> ...

Done.

> \dontrun{} should only be used if the example really cannot be executed
> (e.g. because of missing additional software, missing API keys, ...) by
> the user. That's why wrapping examples in \dontrun{} adds the comment
> ("# Not run:") as a warning for the user.
> Does not seem necessary.
> 
> Please unwrap the examples if they are executable in < 5 sec, or replace
> \dontrun{} with \donttest{}.

The examples cannot be executed directly after the pkg installation
because they need additional python dependencies which first have to be 
installed by the `sjSDM::install_sjSDM()` function. Therefore we deiced to use
the `\dontrun{}` flag

> 
> Please make sure that you do not change the user's options, par or
> working directory. If you really have to do so within functions, please
> ensure with an *immediate* call of on.exit() that the settings are reset
> when the function is exited. e.g.:
> ...
> oldpar <- par(no.readonly = TRUE) # code line i
> on.exit(par(oldpar)) # code line i + 1
> ...
> par(mfrow=c(2,2)) # somewhere after
> ...
> 
> e.g.: sjSDM_cv.R
> If you're not familiar with the function, please check ?on.exit. This
> function makes it possible to restore options before exiting a function
> even if the function breaks. Therefore it needs to be called immediately
> after the option change within a function.

Done. All plot functions have been updated accordingly. 

> 
> Please ensure that you do not modify the global environment (e.g. by
> using <<-) in your functions. This is not allowed by the CRAN policies.

Done. We haved moved global variables into the pkg.env.


### Successfull R CMD checks under

* Locally: MacOS Monterey 12.1 (R x86_64 version)
* Github actions: 
  - MacOS Catalina 10.15 R-release 
  - Ubuntu 20.04 R-release, R-oldrelease, and R-development
  - Windows-latest R-release
* Rhub:
  - fedora 
  - solairs
* Win-builder R-release, R-development, and R-oldrelease

Notes (from win-builder): 

Possibly misspelled words in DESCRIPTION:
  CPUs (21:262)
  GPUs (21:271)
  Hartig (21:320)
  Pichler (21:310)
  Scalable (3:8)
  autocorrelation (21:488)
  jSDMs (21:79, 21:565)
  scalable (21:16)

However, this seems fine to me. The spelling is intended.



### Submission 1, 01/05/2022

This is a new submission. 

Examples are locally tested (runtimes are too high for Rcmdcheck).

Description:

A scalable method to estimate joint Species Distribution Models (jSDMs) for big community datasets based on a Monte Carlo approximation of the joint likelihood.  The numerical approximation is based on 'PyTorch' and 'reticulate', and can be run on CPUs and GPUs alike. The method is described in Pichler & Hartig (2021) <doi:10.1111/2041-210X.13687>. The package contains various extensions, including support for different response families, ability to account for spatial autocorrelation, and deep neural networks instead of the linear predictor in jSDMs.

### Successfull R CMD checks under

* Locally: MacOS Monterey 12.1 (R x86_64 version)
* Github actions: 
  - MacOS Catalina 10.15 R-release 
  - Ubuntu 20.04 R-release, R-oldrelease, and R-development
  - Windows-latest R-release
* Rhub:
  - fedora 
  - solairs
* Win-builder R-release, R-development, and R-oldrelease

Notes (from win-builder): 

Possibly misspelled words in DESCRIPTION:
  CPUs (21:262)
  GPUs (21:271)
  Hartig (21:320)
  Pichler (21:310)
  Scalable (3:8)
  autocorrelation (21:488)
  jSDMs (21:79, 21:565)
  scalable (21:16)

However, this seems fine to me. The spelling is intended. 