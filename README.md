# s-jSDM - Fast and accurate Joint Species Distribution Modeling

For both, the R and the python package, python >= 3.6 is required (python 2 is no longer supported by PyTorch) 

## R-package
```{r}
devtools::install_github("https://github.com/TheoreticalEcology/s-jSDM", subdir = "sjSDM/sjSDM")
```

### Install instructions
```{r}
sjSDM::install_sjSDM(verion = "gpu") # or
sjSDM::install_sjSDM(version = "cpu")
```
  
### Example
```{r}
library(sjSDM)
community <- simulate_SDM(sites = 100, species = 10, env = 5)
Env <- community$env_weights
Occ <- community$response

model <- sjSDM(X = Env, Y = Occ, formula = ~0+X1*X2 + X3 + X4)
summary(model)
```


## Python Package
```{python}
pip install sjSDM_py
```

### Example
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
For details, see [sjSDM_py](https://github.com/TheoreticalEcology/s-jSDM/tree/master/sjSDM/sjSDM-python)


#### MacOS
The PyTorch pip package does not provide CUDA support for MacOS. Refer to [PyTorch](https://pytorch.org/) for install instructions.
