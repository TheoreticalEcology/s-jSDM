# Regression tests for the two backend bugs found by the 2026-09-02 audit of the torch port.
# Both fail against the pre-fix backend_model.R.

testthat::test_that("the environmental penalty skips an intercept-only design", {
  sjSDM:::check_module()
  pen = function(P, intercept) {
    m = sjSDM:::Model_sjSDM(blocks = list(list(input_shape = P, output_shape = 3L, l1 = 0.5,
                                               l2 = -99, intercept = intercept)),
                            loss = list(link = "probit", species = 3L, df = 2L), seed = 42L)
    sjSDM:::set_state(m$env, matrix(rep(seq_len(P), each = 3L), nrow = 3L, ncol = P))
    as.numeric(m$net$penalty()[[1]]$cpu())
  }
  # python penalises p[:,1:], so an intercept-only [3, 1] weight contributes exactly 0.
  # update.sjSDM carries l1/l2 into every ~1 refit, so this reaches get_null_ll() and anova().
  testthat::expect_equal(pen(1L, TRUE), 0.0, tolerance = 1e-5)
  testthat::expect_equal(pen(2L, TRUE), 3.0, tolerance = 1e-5)
  testthat::expect_equal(pen(1L, FALSE), 1.5, tolerance = 1e-5)
})

testthat::test_that("se() reports NaN rather than a plausible number off a maximum", {
  sjSDM:::check_module()
  sim = sjSDM::simulate_SDM(env = 2L, species = 3L, sites = 60L, seed = 1L)
  X = sim$env_weights; colnames(X) = paste0("X", seq_len(2))
  m = sjSDM::sjSDM(sim$response, env = sjSDM::linear(X, ~.), iter = 5L, step_size = 10L,
                   se = FALSE, sampling = 10L, verbose = FALSE)
  # a point far from the optimum: the Hessian is not positive definite there, so inv(H) has
  # negative diagonal entries and sqrt() must be NaN. abs()$sqrt() hid that behind a finite SE.
  sjSDM:::set_state(m$model$env, matrix(20, nrow = 3L, ncol = ncol(m$data$X)))
  se = do.call(rbind, m$model$se(m$data$X, sim$response, sampling = 10L, verbose = FALSE))
  testthat::expect_true(anyNA(se))
})
