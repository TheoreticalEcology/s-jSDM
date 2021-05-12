library(sjSDM)
set.seed(42)
community <- simulate_SDM(sites = 100, species = 10, env = 3)
Env <- community$env_weights
Occ <- community$response

# Full model 
model <- sjSDM(Y = Occ, env = linear(data = Env, formula = ~X1+X2+X3), 
               se = TRUE, family=binomial("probit"), sampling = 100L)

# covariance of the full model
model$sigma

# Open question: how do we get from this to a low-ran approximation of Sigma?
# https://math.stackexchange.com/questions/1863390/decompose-a-matrix-into-diagonal-term-and-low-rank-approximation
# https://rdrr.io/cran/irlba/man/ssvd.html
# https://www.cs.yale.edu/homes/el327/papers/lowRankMatrixApproximation.pdf
# https://en.wikipedia.org/wiki/Low-rank_approximation
# https://stats.stackexchange.com/questions/35209/why-bother-with-low-rank-approximations

# LVM model fit with sjSDM 
lvm = sLVM(Y = Occ, X = Env, lv =2, family = binomial("probit"))
summary(lvm)

# factor loadings
lvm$lf

# construct covariance
t(lvm$lf) %*% lvm$lf
# question Max - does this need another diag(0.5, 3) ?

