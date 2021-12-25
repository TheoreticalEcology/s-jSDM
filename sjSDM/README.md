# sjSDM - Fast and accurate Joint Species Distribution Modeling

A scalable method to estimates joint Species Distribution Models (jSDMs) based on the multivariate probit model through Monte-Carlo approximation of the joint likelihood. The numerical approximation is based on 'PyTorch' and 'reticulate', and can be calculated on CPUs and GPUs alike. 

The method is described in Pichler & Hartig (2021) A new joint species distribution model for faster and more accurate inference of species associations from big community data, https://doi.org/10.1111/2041-210X.13687.

The package includes options to fit various different (j)SDM models:

* jSDMs with Binomial, Poisson and Normal distributed responses
* jSDMs based on deep neural networks
* Spatial eigenvectors or trend surface polynoms can be used to account for spatial auto-correlation

To get more information, install the package and run

```{r}
library(sjSDM)
?sjSDM
vignette("sjSDM", package="sjSDM")
```

**Note**:

sjSDM is based on 'PyTorch', a 'python' library, and thus requires 'python' dependencies. The 'python' dependencies can be automatically installed by running:

```{r}
library(sjSDM)
install_sjSDM()
```
If this didn't work, please check the troubleshooting guide:
```{r}
library(sjSDM)
?installation_help
```