#!/bin/bash
Rscript analysis/gpu_deepjsdm.R > /dev/null 2>&1 & disown
Rscript analysis/cpu_deepjsdm.R > /dev/null 2>&1 & disown
Rscript analysis/gllvm.R > /dev/null 2>&1 & disown
Rscript analysis/hmsc.R > /dev/null 2>&1 & disown
Rscript analysis/BayesComm.R > /dev/null 2>&1 & disown
Rscript analysis/BayesCommDiag.R > /dev/null 2>&1 & disown
Rscript analysis/hmscDiag.R > /dev/null 2>&1 & disown



Rscript analysis/covariance_behaviour.R > /dev/null 2>&1 & disown
Rscript analysis/case_study_1.R > /dev/null 2>&1 & disown
Rscript analysis/large_scale.R > /dev/null 2>&1 & disown