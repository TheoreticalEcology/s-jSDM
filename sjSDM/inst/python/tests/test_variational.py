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
    def _get(inp=5,hidden=5, activation=None, bias=True, df=5, 
             optimizer=fa.optimizer_adamax(1), prior=0.5, prior_trainable=False,l1_cov=0.0,l2_cov=0.0):
        model = fa.Model_base(inp)
        model.add_layer(fa.layers.Layer_dense_variational(hidden=hidden, 
                                                          activation=activation,
                                                          bias=bias,
                                                          prior=0.5,
                                                          prior_trainable=prior_trainable))
        model.build(df=df, l1=l1_cov,l2=l2_cov,optimizer=optimizer)
        return model
    return _get



def test_base(data, model_dense):
    X, Y = data()
    model = model_dense()
    model.fit(X, Y, epochs=1, batch_size=50)

def test_predict(data, model_dense):
    X, Y = data()
    model = model_dense()
    model.fit(X, Y, epochs=1, batch_size=50)
    model.predict(X, batch_size=33)

def test_ll(data, model_dense):
    X, Y = data()
    model = model_dense()
    model.fit(X, Y, epochs=1, batch_size=50)
    model.logLik(X, Y,batch_size=33)


def test_base_2(data, model_dense):
    X, Y = data(10,7)
    model = model_dense(10, 7)
    model.fit(X, Y, epochs=1, batch_size=50)

def test_predict_2(data, model_dense):
    X, Y = data(10,7)
    model = model_dense(10, 7)
    model.fit(X, Y, epochs=1, batch_size=50)
    model.predict(X, batch_size=33)

def test_ll_2(data, model_dense):
    X, Y = data(10,7)
    model = model_dense(10, 7)
    model.fit(X, Y, epochs=1, batch_size=50)
    model.logLik(X, Y,batch_size=33)



def test_base_3(data, model_dense):
    X, Y = data(10,7)
    model = model_dense(10, 7, bias = True)
    model.fit(X, Y, epochs=1, batch_size=50)

def test_predict_3(data, model_dense):
    X, Y = data(10,7)
    model = model_dense(10, 7, bias=True, prior = 3.0)
    model.fit(X, Y, epochs=1, batch_size=50)
    model.predict(X, batch_size=33)

def test_all_1(data, model_dense):
    X, Y = data(10,7)
    model = model_dense(10, 7,bias=True, prior_trainable=True)
    model.fit(X, Y, epochs=1, batch_size=50)
    model.logLik(X, Y,batch_size=33)
    model.predict(X, batch_size=33)

def test_all_2(data, model_dense):
    X, Y = data(10,7)
    model = model_dense(10, 7,bias=True, activation="relu", prior_trainable=True)
    model.fit(X, Y, epochs=1, batch_size=50)
    model.logLik(X, Y,batch_size=33)
    model.predict(X, batch_size=33)

def test_all_3(data, model_dense):
    X, Y = data(10,7)
    model = model_dense(10, 7,bias=True,activation="sigmoid", prior_trainable=True)
    model.fit(X, Y, epochs=1, batch_size=50)
    model.logLik(X, Y,batch_size=50)
    model.predict(X, batch_size=50)

def test_all_3(data, model_dense):
    X, Y = data(10,7)
    model = model_dense(10, 7,bias=True,activation="tanh", prior_trainable=True)
    model.fit(X, Y, epochs=1, batch_size=50)
    model.logLik(X, Y,batch_size=50)
    model.predict(X, batch_size=50)