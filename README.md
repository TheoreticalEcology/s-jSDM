# s-jSDM - Fast and accurate Joint Species Distribution Modeling


Note: for both the R and the python package, python >= 3.6 and pytorch must be installed (more details below).

## R-package

Install the package via

```{r}
devtools::install_github("https://github.com/TheoreticalEcology/s-jSDM", subdir = "sjSDM")
```

Depencies for the package can be installed before or after installing the package. Detailed explanations of the dependencies are provided in vignette("Dependencies", package = "sjSDM"), source code [here](https://github.com/TheoreticalEcology/s-jSDM/blob/master/sjSDM/vignettes/Dependencies.Rmd). Very briefly, If you have conda installed, the dependencies can be automatically installed from within R:

```{r}
sjSDM::install_sjSDM(version = "gpu") # or
sjSDM::install_sjSDM(version = "cpu")
```
Once the dependencies are installed, the following code should run:

```{r}
library(sjSDM)
community <- simulate_SDM(sites = 100, species = 10, env = 5)
Env <- community$env_weights
Occ <- community$response

model <- sjSDM(X = Env, Y = Occ, formula = ~0+X1*X2 + X3 + X4)
summary(model)
```

If it fails, check out the help of ?install_sjSDM and vignette("Dependencies", package = "sjSDM")


## Python Package
```{python}
pip install sjSDM_py
```
Python example

```{python}
import sjSDM_py as fa
import numpy as np
Env = np.random.randn(100, 5)
Occ = np.random.binomial(1, 0.5, [100, 10])

model = fa.Model_base(5)
model.add_layer(fa.layers.Layer_dense(10))
model.build(df=5, optimizer=fa.optimizer_adamax(0.1))
model.fit(X = Env, Y = Occ)
print(model.weights_numpy)
print(model.get_cov())
```
For details, see [sjSDM_py](https://github.com/TheoreticalEcology/s-jSDM/tree/master/sjSDM/python/)
