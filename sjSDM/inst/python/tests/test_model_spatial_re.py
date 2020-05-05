import pytest
from .. import sjSDM_py as fa
import numpy as np

@pytest.fixture
def data():
    def _get(a=5, b=5, c=100, re = 10):
        return (np.random.randn(c,a), np.random.binomial(1,0.5,[c, b]), np.random.randint(0, re, size=[c]))
    return _get


@pytest.fixture
def model_dense():
    def _get(inp=5,hidden=5, activation=None, bias=True, df=5, re = 10,
             optimizer=fa.optimizer_adamax(1), l1_d = 0.0, l2_d = 0.0, 
             l1_cov=0.0, l2_cov=0.0):
        model = fa.Model_spatialRE(inp)
        model.add_layer(fa.layers.Layer_dense(hidden=hidden, 
                                              activation=activation,
                                              bias=bias,
                                              l1=l1_d,
                                              l2=l2_d))
        model.build(df=df, re= 10, l1=l1_cov,l2=l2_cov,optimizer=optimizer)
        return model
    return _get



def test_base(data, model_dense):
    X, Y, Re = data()
    model = model_dense()
    model.fit(X, Y, Re, epochs=1, batch_size=50)

def test_predict(data, model_dense):
    X, Y, Re = data()
    model = model_dense()
    model.fit(X, Y, Re, epochs=1, batch_size=50)
    model.predict(X,Re, batch_size=33)
    model.predict(X, batch_size=33)

def test_ll(data, model_dense):
    X, Y, Re = data()
    model = model_dense()
    model.fit(X, Y, Re, epochs=1, batch_size=50)
    model.logLik(X, Y,Re, batch_size=33)

def test_se(data, model_dense):
    X, Y, Re = data()
    model = model_dense()
    model.fit(X, Y,Re, epochs=1, batch_size=50)
    model.se(X, Y,Re,batch_size=33)
    model.se(X, Y,Re,batch_size=53)
    model.se(X, Y,Re,batch_size=33, sampling=150)


def test_base_2(data, model_dense):
    X, Y, Re = data(10,7, re = 3)
    model = model_dense(10, 7, re = 3)
    model.fit(X, Y, Re, epochs=1, batch_size=50)

def test_predict_2(data, model_dense):
    X, Y, Re = data(10,7, re = 3)
    model = model_dense(10, 7, re = 3)
    model.fit(X, Y, Re, epochs=1, batch_size=50)
    model.predict(X, Re, batch_size=33)
    model.predict(X, batch_size=33)

def test_ll_2(data, model_dense):
    X, Y, Re = data(10,7, re = 4)
    model = model_dense(10, 7, re = 4)
    model.fit(X, Y, Re, epochs=1, batch_size=50)
    model.logLik(X, Y, Re, batch_size=33)
    model.logLik(X, Y, batch_size=33)


def test_reg_1(data, model_dense):
    X, Y, Re = data(10,7)
    model = model_dense(10, 7,l1_d=0.1,l2_d=0.1)
    model.fit(X, Y, Re, epochs=1, batch_size=50)
    model.logLik(X, Y, Re, batch_size=50)
    model.predict(X, Re, batch_size=50)
    model.predict(X, batch_size=50)

def test_reg_2(data, model_dense):
    X, Y, Re = data(10,7)
    model = model_dense(10, 7,l1_cov=0.1,l2_cov=0.1)
    model.fit(X, Y, Re, epochs=1, batch_size=50)
    model.logLik(X, Y, Re, batch_size=50)
    model.logLik(X, Y, batch_size=50)
    model.predict(X, Re, batch_size=50)
    model.predict(X, batch_size=50)

def test_reg_3(data, model_dense):
    X, Y, Re = data(10,7)
    model = model_dense(10, 7,bias=True,l1_cov=0.1,l2_cov=0.1,l1_d=0.1,l2_d=0.1)
    model.fit(X, Y,Re, epochs=1, batch_size=50)
    model.logLik(X, Y, Re, batch_size=50)
    model.predict(X, Re, batch_size=50)
    model.logLik(X, Y, batch_size=50)
    model.predict(X, batch_size=50)