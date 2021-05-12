library(sjSDM)
set.seed(42)
community <- simulate_SDM(sites = 100, species = 10, env = 3)
Env <- community$env_weights
Occ <- community$response

# Full model 
model <- sjSDM(Y = Occ, env = linear(data = Env, formula = ~X1+X2+X3), 
               se = TRUE, family=binomial("probit"), sampling = 100L)

# covariance of the full model
getCov(model)

# Open question: how do we get from this to a low-ran approximation of Sigma?
# https://math.stackexchange.com/questions/1863390/decompose-a-matrix-into-diagonal-term-and-low-rank-approximation
# https://rdrr.io/cran/irlba/man/ssvd.html
# https://www.cs.yale.edu/homes/el327/papers/lowRankMatrixApproximation.pdf
# https://en.wikipedia.org/wiki/Low-rank_approximation
# https://stats.stackexchange.com/questions/35209/why-bother-with-low-rank-approximations

# LVM model fit with sjSDM 
lvm = sLVM(Y = Occ, X = Env, lv =2, family = binomial("probit"))
summary(lvm)

# latent variables - 'unobserved environmental predictors' (which are used for the ordination in the LVM models)
# M question: I agree with you Florian that it is possible to estimate the best LF matrix for Sigma post-hoc, but what about the latent variables?
# M    We could use the recovered LF and the rest of the model to estimate LV then?
lvm$lv

# factor loadings
lvm$lf

# construct covariance
t(lvm$lf) %*% lvm$lf
# question Max - does this need another diag(0.5, 3) ?
#   M: Wilkinson et al. 2019 says L*t(L) + I but Warton 2015 didn't mention it.
#   M: Gllvm and Hmsc don't add I but they always speak of Sigma as 'residual' correlation/covariance matrix
#   M: But the idea of the low-rank approximation is that we that we separate the contribution of the variances (I) and the covariances ('residual')
#   M:    see first answer from https://math.stackexchange.com/questions/2848517/sampling-multivariate-normal-with-low-rank-covariance 
#   M: Therefore I think that the addition of I is necessary


