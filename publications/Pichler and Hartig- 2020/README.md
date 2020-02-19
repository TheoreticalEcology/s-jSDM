# Pichler and Hartig, A new method for faster and more accurate inference of species associations from novel community data 

This subfolder contains the code to reproduce the results in Pichler and Hartig, A new method for faster and more accurate inference of species associations from novel community data 

## Runtime benchmark
### All models, different simulation scenarios
Generate data:
```{r}
source("./analysis/scripts/1_generate_data.R")
```
Benchmark models:
```{r}
source("analysis/scripts/1_cpu_sjSDM.R") # for s-jSDM on the CPU
source("analysis/scripts/1_gpu_sjSDM.R") # for s-jSDM on the GPU
source("analysis/scripts/1_gllvm.R) # GLLVM package
source("analysis/scripts/1_BayesCommDiag.R") # BayesComm package
source("analysis/scripts/1_hmscDiag.R") # Hmsc package
```
For each model and scenario, the following is calculated:
* Species-species covariance matrix: accuracy, rmse
* Beta (species-env response): accuracy, rmse

### s-jSDM on large scale data
```{r}
source("analysis/scripts/2_large_scale.R")
```
### s-jSDM on Wilkinson et al 2019 datasets:
```{r}
source("analysis/scripts/4_case_study_1.R")
```

## Inference
### Non sparse species species associations
Results are taken from the runtime benchmark (see above)

### Covariance behaviour (for Appendix)
```{r}
source("analysis/scripts/3_covariance_behaviour.R")
```
### Sparse species-species assocations

Generate data:
```{r}
source("analysis/scripts/6_generate_data.R")
```

```{r}
source("analysis/scripts/6_sparse_gpu_sjSDM.R") # for s-jSDM on the GPU
source("analysis/scripts/6_sparse_gllvm.R) # GLLVM package
source("analysis/scripts/6_sparse_bc.R") # BayesComm package
source("analysis/scripts/6_sparse_hmsc.R") # Hmsc package
```

### eDNA Fungi Dataset
```{r}
source("analysis/scripts/5_Fungi_eDNA_analysis.R")
```
