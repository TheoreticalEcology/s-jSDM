# s-jSDM - Fast and accurate Joint Species Distribution Modeling

For both, the R and the python package, python >= 3.6 is required (python 2 is no longer supported by PyTorch) 

## R-package
```{r}
devtools::install_github("https://github.com/TheoreticalEcology/s-jSDM", subdir = "sjSDM")
```

If you have conda installed, the dependencies can be automatically installed from within R:
```{r}
sjSDM::install_sjSDM(version = "gpu") # or
sjSDM::install_sjSDM(version = "cpu")
```
Try to run the [example](#example). If it fails, check out the section below.


## Installation
* [Windows](#windows)
* [Linux](#linux)
* [MacOS](#macos)


### Windows
Conda is the easiest way to install python and python packages on windows:
- Install the latest [conda version](https://www.anaconda.com/distribution/)
- Open the command window (cmd.exe - hit windows key + r and write cmd)

Run:
```
$ conda create --name sjSDM_env python=3.7
$ conda activate sjSDM_env
$ conda install pip
$ conda install pytorch torchvision cpuonly -c pytorch # cpu
$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch #gpu
```



### Linux
#### Pip
python3 is pre-installed on most linux distributions, but you have to check that the minimal requirement of python >= 3.6 is met: 

```
$ python3 --version 
$ python --version
```

Install pip
```
$ sudo apt install python3-pip # for ubuntu/deb
```

Create a virtualenv and install dependencies:
```
$ python3 -m pip install --user virtualenv
$ python3 -m venv ~/sjSDM_env
$ source ~/sjSDM_env/bin/activate
$ pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html #cpu
$ pip install torch torchvision #gpu
```

start RStudio from within the virtualenv and try the [example](#example)



#### Conda
Install the latest [conda version](https://www.anaconda.com/distribution/) and run:
```
$ conda create --name sjSDM_env python=3.7
$ conda activate sjSDM_env
$ conda install pip
$ conda install pytorch torchvision cpuonly -c pytorch # cpu
$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch #gpu
```
start RStudio from  within the conda env and try the [example](#example)



### MacOS
#### Conda
Install the latest [conda version](https://www.anaconda.com/distribution/) and run:
```
$ conda create --name sjSDM_env python=3.7
$ conda activate sjSDM_env
$ conda install pip
$ conda install pytorch torchvision cpuonly -c pytorch # cpu
```
start RStudio from  within the conda env and try the [example](#example)

For GPU support on MacOS, you have to install the cuda binaries yourself, see [PyTorch for help](https://pytorch.org/)

  
## Example
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

## Python example
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
