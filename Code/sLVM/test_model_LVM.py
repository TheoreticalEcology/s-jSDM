import pytest
from .. import sjSDM_py as fa
import numpy as np

@pytest.fixture
def data():
    def _get(a=5, b=5, c=100):
        return (np.random.randn(c,a), np.random.binomial(1,0.5,[c, b]))
    return _get


@pytest.fixture
def model_base():
    def _get(X, Y,df=2, guide="Delta", 
            scale_mu = 5.0, scale_lf = 1.0,scale_lv=1.0, 
            lr = [0.1], 
            epochs = 2,
            family = "binomial",
            link = "logit",
            batch_size = 25,
            num_samples=100):
        model = fa.Model_LVM()
        model.build(X.shape, Y.shape[1], df=df,guide=guide,
                  scale_mu=scale_mu,scale_lf=scale_lf,scale_lv=scale_lv, family=family,link=link)
        model.fit(X=X, Y=Y,epochs=epochs,batch_size=batch_size, num_samples=num_samples)
        return model
    return _get


@pytest.mark.parametrize("e", [2, 3, 5])
@pytest.mark.parametrize("sp", [3, 6, 11])
@pytest.mark.parametrize("n", [35, 50, 77])
@pytest.mark.parametrize("df", [2,3,5])
@pytest.mark.parametrize("bs", [2, 22])
def test_base(data, model_base, e, sp, n, df, bs):
    X, Y = data(e, sp, n)
    model = model_base(X, Y, df=df, batch_size=bs)
    model.getLogLik(X, Y, batch_size=bs)
    pred = model.predict(X, batch_size=bs)
    assert pred.shape[0] == X.shape[0]


@pytest.mark.parametrize("fam,link", [("binomial", "logit"), ("binomial", "probit"), ("poisson", "log"), ("poisson", "linear")])
def test_fams(data, model_base, fam, link):
    X, Y = data()
    model = model_base(X, Y, family=fam, link=link)
    model.getLogLik(X, Y)
    pred = model.predict(X)
    assert pred.shape[0] == X.shape[0]

@pytest.mark.parametrize("guide", ["LaplaceApproximation", "Delta", "LowRankMultivariateNormal", "DiagonalNormal"])
@pytest.mark.parametrize("lr", [ [0.1], [0.1, 0.1, 0.1] ])
@pytest.mark.parametrize("scales", [(3.0, 1.0, 1.0), (5.0, 3.0, 0.001)])
def test_guide_prior_lrs(data, model_base, guide, lr, scales):
    X, Y = data()
    model = model_base(X, Y, guide=guide, lr=lr, scale_mu=scales[0], scale_lf=scales[1], scale_lv=scales[2])
    model.getLogLik(X, Y)
    pred = model.predict(X)
    assert pred.shape[0] == X.shape[0]   