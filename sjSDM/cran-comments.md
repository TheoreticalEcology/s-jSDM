## Version 1.0.0

### Submission 1, 01/0.5/2022

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