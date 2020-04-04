import pytest
from .. import sjSDM_py as fa
import numpy as np

@pytest.fixture
def data():
    def _get(a=5, b=5, c=100):
        return (np.random.randn(c,a), np.random.binomial(1,0.5,[c, b]))
    return _get


@pytest.fixture
def model_dense():
    def _get(inp=5,hidden=5, n=2,activation=None, bias=True, df=5, 
             optimizer=fa.optimizer_adamax(1), l1_d = 0.0, l2_d = 0.0, 
             l1_cov=0.0, l2_cov=0.0):
        model = fa.Model_base(inp)
        for _ in range(n):
            model.add_layer(fa.layers.Layer_dense(hidden=hidden, 
                                                activation=activation,
                                                bias=bias,
                                                l1=l1_d,
                                                l2=l2_d))
        model.build(df=df, l1=l1_cov,l2=l2_cov,optimizer=optimizer)
        return model
    return _get



def test_base_1(data, model_dense):
    X, Y = data()
    model = model_dense()
    model.fit(X, Y, epochs=1, batch_size=50)

def test_all_1(data, model_dense):
    X, Y = data()
    model = model_dense()
    model.fit(X, Y, epochs=1, batch_size=50)
    model.predict(X, batch_size=33)
    model.logLik(X, Y, batch_size=50)

def test_all_2(data, model_dense):
    X, Y = data()
    model = model_dense(n=4)
    assert len(model.layers) == 4
    model.fit(X, Y, epochs=1, batch_size=50)
    model.predict(X, batch_size=33)
    model.logLik(X, Y, batch_size=50)

def test_all_3(data, model_dense):
    X, Y = data(12,5)
    model = model_dense(12,5, activation="relu")
    model.fit(X, Y, epochs=1, batch_size=50)
    model.predict(X, batch_size=33)
    model.logLik(X, Y, batch_size=50)

def test_all_4(data, model_dense):
    X, Y = data(12,5)
    model = model_dense(12,5, activation="relu",bias=True)
    model.fit(X, Y, epochs=1, batch_size=50)
    model.predict(X, batch_size=33)
    model.logLik(X, Y, batch_size=50)

def test_all_5(data, model_dense):
    X, Y = data(12,5)
    model = model_dense(12,5, activation="relu", l1_d=0.5, l2_d=0.5)
    model.fit(X, Y, epochs=1, batch_size=50)
    model.predict(X, batch_size=33)
    model.logLik(X, Y, batch_size=50)

def test_all_6(data, model_dense):
    X, Y = data(12,5)
    model = model_dense(12,5, activation="tanh", l1_d=0.5, l2_d=0.5,l1_cov=0.1)
    model.fit(X, Y, epochs=1, batch_size=50)
    model.predict(X, batch_size=33)
    model.logLik(X, Y, batch_size=50)

def test_all_7(data, model_dense):
    X, Y = data(12,5)
    model = model_dense(12,5, activation="tanh", l1_d=0.5, l2_d=0.5,l1_cov=0.1, l2_cov=0.1)
    model.fit(X, Y, epochs=1, batch_size=50)
    model.predict(X, batch_size=33)
    model.logLik(X, Y, batch_size=50)


def test_all_8(data, model_dense):
    X, Y = data(12,50)
    model = model_dense(12,50,n=3, activation="tanh", l1_d=0.5, l2_d=0.5,l1_cov=0.1, l2_cov=0.1,df=10)
    model.fit(X, Y, epochs=1, batch_size=50)
    model.predict(X, batch_size=33)
    model.logLik(X, Y, batch_size=50)