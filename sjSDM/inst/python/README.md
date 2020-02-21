# sjSDM - Fast and Accurate Joint Species distribution Modeling 

At the moment, we do not provide specifically a API for joint species distribution models. However, it's just a deep multivariate probit model with one layer (example below).

We provide a R package with an API focused on jSDM is available [here](https://github.com/TheoreticalEcology/s-jSDM).

## Install instructions

Dependencies:
* PyTorch >= 1.4, see [PyTorch](https://pytorch.org/get-started/locally/) for install instructions.

```{python}
pip install sjSDM_py
```

## Example

linear jSDM:
```{python}
import sjSDM_py as sa
import numpy as np
Env = np.random.randn(100, 5)
Occ = np.random.binomial(1, 0.5, [100, 10])

model = sa.Model_base(5) # input_shape == number of environmental predictors
model.add_layer(sa.layers.Layer_dense(hidden=10)) # number of hidden units in the layer == number of species
model.build(df=5, optimizer=sa.optimizer_adamax(lr=0.1, weight_decay = 0.01)) # df = degree of freedom 
model.fit(X = Env, Y = Occ)
print(model.weights_numpy)
print(model.get_cov())
```

* For species intercept, use 'bias=True' in 'Layer_dense(...)'. 
* We recommend to set 'df = number of species / 2.'

