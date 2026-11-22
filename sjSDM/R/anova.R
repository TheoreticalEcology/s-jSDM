#' Anova / Variation partitioning
#' 
#' Compute variance explained by the three fractions env, space, associations
#' 
#' @param object model of object \code{\link{sjSDM}}
#' @param samples Number of Monte Carlo samples
#' @param proposal distribution the latent factors are drawn from. `"laplace"` (the default) draws
#'   from a Gaussian fitted to the per-site mode of the integrand and reweights the draws.
#'   `"prior"` draws from `N(0, I)`, which is what every version before 1.1.0 did. The reweighting keeps
#'   the estimator of the likelihood unbiased and raises the effective sample size by orders of
#'   magnitude at many species. The log of that estimate, which is what the fractions are built
#'   from, keeps a bias of roughly `(1/ess - 1)/(2 * samples)` per site. See Details.
#' @param verbose `TRUE` or `FALSE`, indicating whether progress should be printed or not
#' @param ... optional arguments which are passed to the calculation of the logLikelihood,
#'   including \code{mc_batch}, the number of sites whose Monte-Carlo draws are held in
#'   memory at once. It defaults to the model's \code{step_size}, lowered if that would
#'   exceed roughly 2.5 GB; lower it by hand on a device with less memory. For a model with
#'   random effects also \code{re_draws} (default 50), the number of draws of the group effects
#'   their marginal likelihood is averaged over; the cost of the anova grows linearly with it.
#' 
#' @details The ANOVA function removes each of the three fractions (Environment, Space, Associations) and measures the drop in variance explained, and thus the importance of the three fractions.
#' 
#' Variance explained is measured by Deviance as well as the pseudo-R2 metrics of Nagelkerke and McFadden
#' 
#' In downstream functions such as \code{\link{plot.sjSDManova}} or \code{\link{plot.sjSDManova}} with \code{add_shared=TRUE}.
#' The anova can get unstable for many species and few occurrences/observations. We recommend using large numbers for 'samples'.
#' 
#' The sub-models without associations, and the null model, carry no latent term at all. Up to 1.0.7 they
#' were fitted with an identity covariance, which for a probit link only rescales the coefficients but for
#' the count, negative binomial and gaussian links adds a fixed unit latent variance; that misfit was then
#' reported as associations.
#'
#' For the count family this has a consequence worth knowing: the latent term of the full model also
#' carries each species' own extra variance (the diagonal of the association matrix), i.e. log-normal
#' overdispersion, and the sub-models without associations cannot. Overdispersion is therefore part of the
#' Associations fraction. In simulations with 7 independent species and 180 sites, the Associations
#' fraction (McFadden R-squared, mean over 5 datasets) of a Poisson model was 0.002 without
#' overdispersion, 0.02 with a log-normal standard deviation of 0.5 and 0.15 with 1. For overdispersed
#' counts use \code{family = "nbinom"}, whose dispersion parameter is present in every sub-model; on
#' the same data it was 0.002, 0.004 and 0.008.
#' 
#' `proposal = "prior"` draws the latent factors blind and weights them by the likelihood.
#' At many species almost every draw lands where the observed community is implausible, so the effective
#' sample size collapses -- measured at 100 species it is 3 draws out of 1000 -- and the negative
#' log-likelihood \code{-log(mean(exp(lp)))} is biased upwards by an amount that grows with the variance
#' of the per-draw log-probability. That variance differs between the sub-models \code{anova} compares,
#' so it does not cancel in the differences the fractions are built from, and the sub-model that leans
#' hardest on the latent factors is depressed most. `proposal = "laplace"` reduces this by sampling from
#' a Gaussian centred on the per-site mode with the curvature of the integrand, and correcting each draw
#' by \code{log p(z) - log q(z)}. It is a proposal, not an approximation: a poor fit costs effective
#' sample size, not correctness. The realised fraction is returned in the `ess` component, where 1 means
#' every draw counted. Read it as a per-dimension efficiency raised to the power of the latent dimension
#' `df`, not as a per-site quality score. The sub-models without associations have nothing to integrate
#' and report 1.
#' 
#' For the gaussian family the McFadden R-squared is a ratio of log-densities, which change with the
#' measurement scale of the response and turn positive for small residual variances, so it is not bounded
#' by 1 and differs between units. The deviances and the Nagelkerke R-squared do not have this problem.
#' 
#' \code{anova} is not defined for a model fitted with \code{bioticStruct(diag = TRUE)}, whose sub-models with
#' associations would not be nested in it, and raises an error.
#' 
#' For a model with random-effect bars the anova is conditional on the random intercepts: the null model
#' and every sub-model contain the random intercept of each grouping factor, so the fractions are the
#' variation explained by environment, associations and space beyond the grouping. A random slope belongs
#' to the abiotic component and is removed with it; for \code{(temp | plot)} the null holds
#' \code{(1 | plot)} and only the sub-models with the abiotic component hold \code{(temp | plot)}. The
#' \code{grouping} component reports what the random intercepts alone explain against an intercept-only
#' model. Every model is scored on its marginal likelihood, with the group effects integrated out by
#' importance sampling from their fitted posterior (\code{re_draws}, see \code{...}). With a random
#' intercept on a site or plot identifier, covariates that are constant within sites compete with it for
#' the variation between groups: in simulations such a covariate kept about half of its deviance, while
#' covariates that vary within groups kept all of it. The estimated group variance does not reach zero,
#' so a random intercept without a true group effect still takes a little from the full model (about
#' 2.5 per cent of its R-squared in simulations), and a random slope without a true slope variance adds a
#' little to the abiotic fraction. With crossed grouping factors, e.g. \code{(1 | plot) + (1 | year)},
#' the group effects cannot be integrated separately and \code{re_draws} has to be raised substantially.
#' 
#' The \code{species} and \code{sites} components are not the printed table restricted to a species or a
#' site, and they are read differently. The table composes the fractions on the log-likelihood scale and
#' forms a single ratio at the end, so it is exact inclusion-exclusion on the joint likelihoods. The
#' per-species and per-site values are allocations: each site's improvement over the null is apportioned
#' across species in proportion to how much each species' conditional likelihood improved, and that share
#' is computed separately for every sub-model. A per-species or per-site fraction is therefore a difference
#' of two allocations computed under different shares, it is unbounded in both directions, and 1 is not a
#' boundary for it. The shares are undefined when a sub-model is about as good as the null overall, so
#' treat per-species and per-site values from such a model with care. The site totals are unaffected:
#' the shares cancel when summed over species.
#' 
#' @return 
#' An S3 class of type 'sjSDManova' including the following components:
#' 
#' \item{results}{Data frame of results.}
#' \item{to_print}{Data frame, summarized results for type I anova.}
#' \item{N}{Number of observations (sites).}
#' \item{spatial}{Logical, spatial model or not.}
#' \item{species}{individual species R2s.}
#' \item{sites}{individual site R2s.}
#' \item{lls}{individual site by species negative-log-likelihood values.}
#' \item{proposal}{the sampler the latent factors were drawn from.}
#' \item{grouping}{for a model with random effects: log-likelihood of the intercept-only model, of the
#'   null with random intercepts, and the McFadden R-squared of the random intercepts alone.}
#' \item{ess}{sites-by-sub-model matrix of effective sample sizes, as a fraction of `samples`.
#'   One column per fitted sub-model, so a low column shows which model's likelihood rests on few
#'   draws. `attr(, "newton_decrement")` carries the largest Newton decrement the Laplace mode-finder left
#'   behind, per sub-model.}
#' \item{object}{the fitted model the anova was computed for}
#' 
#' Implemented S3 methods are \code{\link{print.sjSDManova}} and \code{\link{plot.sjSDManova}}
#'  
#' @seealso \code{\link{plot.sjSDManova}}, \code{\link{print.sjSDManova}},\code{\link{summary.sjSDManova}}, \code{\link{plot.sjSDMinternalStructure}}
#' 
#' @example /inst/examples/anova-example.R
#' 
#' @import stats
#' @export

anova.sjSDM = function(object, samples = 5000L, verbose = TRUE,
                       proposal = c("laplace", "prior"), ...) {
  out = list()
  individual = TRUE
  samples = as.integer(samples)
  proposal = match.arg(proposal)
  object = checkModel(object)
  # with a diagonal covariance the association-carrying sub-models would not be nested in it
  if(isTRUE(object$settings$biotic$diag))
    stop("anova() needs a model with associations, it is not defined for ",
         "bioticStruct(diag = TRUE)", call. = FALSE)
  object$samples = samples
  # `samples` is the Monte-Carlo budget for EVALUATING the fractions, and it is passed
  # explicitly to every get_conditional_lls() call below. It must not reach the refits:
  # update() hands settings$sampling to sjSDM(), so writing it here trained the seven
  # sub-models and the null at `samples` draws per optimiser step while full_m came from
  # the original fit at the model's own sampling -- the fractions then compared models
  # trained under different regimes, and a 201-species anova spent its time there.
  
  sjsdm_set_seed(object$seed)
  #   if(object$family$family$family == "gaussian") stop("gaussian not yet supported")
  
  object$settings$se = FALSE
  
  # random intercepts belong to the base: the null and every sub-model carry them, so the
  # fractions are conditional on the grouping; random slopes go with their predictor (Abiotic)
  base = anova_ri_formula(object)
  null_m = -get_null_ll(object, verbose = verbose, condition = TRUE, sampling = samples,
                        proposal = proposal, ...)
  # a model whose only bars are random slopes has no grouping row: its base is ~1
  if(any(vapply(object$settings$env$re, function(b) isTRUE(b$intercept), TRUE))) {
    pure = get_null_ll(object, verbose = verbose)
    out$grouping = c(ll_intercept = sum(pure), ll_grouping = -sum(null_m),
                     R2_McFadden = 1 - sum(null_m) / -sum(pure))
  }
  
  full_m = get_conditional_lls(object, null_m, sampling = samples, proposal = proposal, ...)
  
  ### fit different models ###
  #e_form = stats::as.formula(paste0(as.character(object$settings$env$formula), collapse = ""))
  if(!inherits(object, "spatial")) {
    out$spatial = FALSE
    
    m = update(object, env_formula = NULL, spatial_formula = ~0, biotic = anova_biotic(object, TRUE), verbose = verbose)
    A_m = get_conditional_lls(m, null_m, sampling = samples, proposal = proposal, ...)
    m = update(object, env_formula = base, spatial_formula = ~0, biotic = anova_biotic(object, FALSE), verbose = verbose)
    B_m = get_conditional_lls(m, null_m, sampling = samples, proposal = proposal, ...)
    SAT_m = saturated_nll(object$data$Y, object$family$link, object$theta)
    
    A_wo = A_m - null_m
    B_wo = B_m - null_m
    full_wo = full_m - null_m
    
    F_A = full_wo - B_wo
    F_B = full_wo - A_wo
    F_AB = full_wo - F_A - F_B
    
    
    R2object = list(A = sum(A_m), B = sum(B_m), AB = sum(full_m - A_m - B_m), Full = sum(full_m))
    R2object = lapply(R2object, function(l) l - sum(null_m))
    R2s = get_shared_anova(R2object, "2")
    
    results_discard = data.frame(models = c("F_A", "F_B","F_AB","Full", "Saturated", "Null"),
                                 ll = -c(sum(null_m) + sum(F_A), sum(null_m)  + sum(F_B), sum(null_m)  + sum(F_AB), sum(full_m), sum(SAT_m), sum(null_m)))
    
    
    results_proportional = data.frame(models = c("F_A", "F_B","F_AB","Full", "Saturated", "Null"),
                                      ll = -c(sum(null_m) + sum(R2s$proportional$F_A, na.rm = TRUE), 
                                              sum(null_m) + sum(R2s$proportional$F_B, na.rm = TRUE), sum(null_m)  + sum(F_AB), sum(full_m), sum(SAT_m), sum(null_m)))
    
    results_equal = data.frame(models = c("F_A", "F_B","F_AB","Full", "Saturated", "Null"),
                               ll = -c(sum(null_m) + sum(R2s$equal$F_A, na.rm = TRUE), 
                                       sum(null_m) + sum(R2s$equal$F_B, na.rm = TRUE), sum(null_m)  + sum(F_AB), sum(full_m), sum(SAT_m), sum(null_m)))
    
    # mvp = list(F_A = R2objt$A, F_B = R2objt$Full - R2objt$A, F_S = S, R2 = R2objt$Full)
    # mvp_shared = mvp
    
    results_mvp = data.frame(models = c("F_A", "F_B", "Full", "Saturated", "Null"),
                                      ll = -c(sum(null_m) + sum(R2s$proportional$F_A, na.rm = TRUE), 
                                              sum(null_m) + sum(R2s$proportional$F_B, na.rm = TRUE),sum(full_m), sum(SAT_m), sum(null_m)))
    
    results_mvp_proportional = results_mvp
    
    results_ind = list("F_A"=-(null_m + F_A), "F_B"=-(null_m +F_B), "F_AB"=-(null_m + F_AB), "A" = -A_m, "B" = -B_m, "Full"=-full_m, "Saturated"=-SAT_m, "Null"=-null_m)
    out$ess = collect_ess(list(A = A_m, B = B_m, Full = full_m))
    
  } else {
    out$spatial = TRUE
    
    s_form = stats::as.formula(paste0(as.character(object$settings$spatial$formula), collapse = ""))
    # A only model
    m = update(object, env_formula = NULL, spatial_formula = ~0, biotic = anova_biotic(object, TRUE), verbose = verbose)
    A_m = get_conditional_lls(m, null_m, sampling = samples, proposal = proposal, ...)
    
    # B only model
    m = update(object, env_formula = base, spatial_formula = ~0, biotic = anova_biotic(object, FALSE), verbose = verbose)
    B_m = get_conditional_lls(m, null_m, sampling = samples, proposal = proposal, ...)
    
    # S only model
    m = update(object, env_formula = base, spatial_formula = NULL, biotic = anova_biotic(object, TRUE), verbose = verbose)
    S_m = get_conditional_lls(m, null_m, sampling = samples, proposal = proposal, ...)
    
    # AB
    m = update(object, env_formula = NULL, spatial_formula = ~0, biotic = anova_biotic(object, FALSE), verbose = verbose)
    AB_m = get_conditional_lls(m, null_m, sampling = samples, proposal = proposal, ...)
    
    # AS
    m = update(object, env_formula = NULL, spatial_formula = NULL, biotic = anova_biotic(object, TRUE), verbose = verbose)
    AS_m = get_conditional_lls(m, null_m, sampling = samples, proposal = proposal, ...)
    
    # BS
    m = update(object, env_formula = base, spatial_formula = NULL, biotic = anova_biotic(object, FALSE), verbose = verbose)
    BS_m = get_conditional_lls(m, null_m, sampling = samples, proposal = proposal, ...)
    
    # Saturated, do we really need it?
    SAT_m = saturated_nll(object$data$Y, object$family$link, object$theta)
    
    
    A_wo = A_m - null_m
    B_wo = B_m - null_m
    S_wo = S_m - null_m
    AB_wo = AB_m - null_m
    AS_wo = AS_m - null_m
    BS_wo = BS_m - null_m
    full_wo = full_m - null_m
    
    F_A = full_wo - BS_wo
    F_B = full_wo - AS_wo
    F_S = full_wo - AB_wo
    F_AS = full_wo - B_wo - F_A - F_S
    F_AB = full_wo - S_wo - F_A - F_B
    F_BS = full_wo - A_wo - F_S - F_B
    F_ABS = full_wo - F_BS - F_AB- F_AS- F_A- F_B - F_S
    
    R2object = list(A = sum(A_m), B = sum(B_m), S = sum(S_m), AB = sum(AB_m), BS = sum(BS_m), AS = sum(AS_m), Full = sum(full_m))
    R2object = lapply(R2object, function(l) l - sum(null_m))
    R2s = get_shared_anova(R2object, "3")
    
    results_discard = data.frame(models = c("F_A", "F_B","F_S","F_AB","F_AS", "F_BS", "F_ABS", "Full", "Saturated", "Null"),
                                 ll = -c(sum(null_m) + sum(F_A), sum(null_m) + sum(F_B),sum(null_m) + sum(F_S), 
                                         sum(null_m) + sum(F_AB), sum(null_m) + sum(F_AS), sum(null_m) + sum(F_BS), sum(null_m) + sum(F_ABS), 
                                         sum(null_m) + sum(full_wo), sum(SAT_m), sum(null_m)))
    ## proportional
    results_proportional = data.frame(models = c("F_A", "F_B","F_S","F_AB","F_AS", "F_BS", "F_ABS", "Full", "Saturated", "Null"),
                                      ll = -c(sum(null_m) + sum(R2s$proportional$F_A, na.rm = TRUE), 
                                              sum(null_m) + sum(R2s$proportional$F_B, na.rm = TRUE),
                                              sum(null_m) + sum(R2s$proportional$F_S, na.rm = TRUE), 
                                              sum(null_m) + sum(F_AB), sum(null_m) + sum(F_AS), sum(null_m) + sum(F_BS), sum(null_m) + sum(F_ABS), 
                                              sum(null_m) + sum(full_wo), sum(SAT_m), sum(null_m)))
    
    results_mvp = data.frame(models = c("F_A", "F_B","F_S","F_AB","F_AS", "F_BS", "F_ABS", "Full", "Saturated", "Null"),
                             ll = -c(sum(null_m) + sum(R2s$mvp$F_A, na.rm = TRUE), 
                                     sum(null_m) + sum(R2s$mvp$F_B, na.rm = TRUE),
                                     sum(null_m) + sum(R2s$mvp$F_S, na.rm = TRUE), 
                                     sum(null_m) + NA, 
                                     sum(null_m) + sum(R2object$AS-R2s$mvp$F_A-R2s$mvp$F_S, na.rm = TRUE), 
                                     sum(null_m) + NA, 
                                     sum(null_m) + NA, 
                                     sum(null_m) + sum(full_wo), 
                                     sum(SAT_m), 
                                     sum(null_m)))
    
    results_mvp_proportional = data.frame(models = c("F_A", "F_B","F_S","F_AB","F_AS", "F_BS", "F_ABS", "Full", "Saturated", "Null"),
                                          ll = -c(sum(null_m) + sum(R2s$mvp_proportional$F_A, na.rm = TRUE), 
                                                  sum(null_m) + sum(R2s$mvp_proportional$F_B, na.rm = TRUE),
                                                  sum(null_m) + sum(R2s$mvp_proportional$F_S, na.rm = TRUE), 
                                                  sum(null_m) + NA, 
                                                  sum(null_m) + NA, 
                                                  sum(null_m) + NA, 
                                                  sum(null_m) + NA, 
                                                  sum(null_m) + sum(full_wo), 
                                                  sum(SAT_m), 
                                                  sum(null_m)))
    
    
    results_equal = data.frame(models = c("F_A", "F_B","F_S","F_AB","F_AS", "F_BS", "F_ABS", "Full", "Saturated", "Null"),
                               ll = -c(sum(null_m) + sum(R2s$equal$F_A, na.rm = TRUE), 
                                       sum(null_m) + sum(R2s$equal$F_B, na.rm = TRUE),
                                       sum(null_m) + sum(R2s$equal$F_S, na.rm = TRUE), 
                                       sum(null_m) + sum(F_AB), sum(null_m) + sum(F_AS), sum(null_m) + sum(F_BS), sum(null_m) + sum(F_ABS), 
                                       sum(null_m) + sum(full_wo), sum(SAT_m), sum(null_m)))
    results_ind = list("F_A"=-(null_m + F_A), "F_B"=-(null_m + F_B),"F_S"=-(null_m +F_S), "F_AB"=-(null_m +F_AB),"F_AS"=-(null_m + F_AS), 
                       "F_BS"=-(null_m + F_BS), "F_ABS"=-(null_m + F_ABS), "A" = -A_m, "B" = -B_m, "S" = -S_m, AB = -AB_m, AS = -AS_m, BS = -BS_m,
                       "Full"=-(full_m), "Saturated"= -(SAT_m), "Null"=-null_m)
    
    out$ess = collect_ess(list(A = A_m, B = B_m, S = S_m, AB = AB_m, AS = AS_m,
                               BS = BS_m, Full = full_m))
    anova_rows = c("Null", "F_A", "F_B", "F_S", "Full")
    names(anova_rows) = c("Null", "Abiotic", "Associations", "Spatial", "Full")
  }
  results = 
    lapply(list(results_discard, results_proportional, results_equal, results_mvp, results_mvp_proportional), function(res) {
      
      res$`Residual deviance` = -2*(res$ll - res$ll[which(res$models == "Saturated", arr.ind = TRUE)])
      
      res$Deviance = res$`Residual deviance`[which(res$models == "Null", arr.ind = TRUE)] - res$`Residual deviance`
      R21 = function(a, b) return(1-exp(2/(nrow(object$data$Y))*(-a+b)))
      res$`R2 Nagelkerke` = R21(rep(-res$ll[which(res$models == "Null", arr.ind = TRUE)], length(res$ll)), - res$ll)
      R22 = function(a, b) 1 - (b/a)
      res$`R2 McFadden`= R22(rep(res$ll[which(res$models == "Null", arr.ind = TRUE)], length(res$ll)), res$ll)
      return(res)
      
    })
  
  # individual
  Residual_deviance_ind = lapply(results_ind, function(r) -2 * (r - results_ind$Saturated))
  Deviance_ind = lapply(Residual_deviance_ind, function(r) Residual_deviance_ind$Null - r)
  
  R211 = function(a, b, n=1) return(1-exp(2/(n)*(-a+b)))   # divide by what?
  R2_Nagelkerke_ind = lapply(results_ind, function(r) R211(-colSums(results_ind$Null), -colSums(r), n=nrow(object$data$Y)))
  R2_Nagelkerke_sites = lapply(results_ind, function(r) R211(-rowSums(results_ind$Null), -rowSums(r), n=ncol(object$data$Y)))
  
  R222 = function(a, b) 1 - (b/a)
  R2_McFadden_ind = lapply(results_ind, function(r) R222(colSums(results_ind$Null), colSums(r)))
  R2_McFadden_sites = lapply(results_ind, function(r) R222(rowSums(results_ind$Null), rowSums(r)))
  
  if( out$spatial ) fractions = "3"
  else fractions = "2"
  
  R2_McFadden_ind_shared = get_shared_anova(R2_McFadden_ind, fractions = fractions)
  R2_McFadden_sites_shared = get_shared_anova(R2_McFadden_sites, fractions = fractions)
  
  R2_Nagelkerke_ind_shared = get_shared_anova(R2_Nagelkerke_ind, fractions = fractions)
  R2_Nagelkerke_sites_shared = get_shared_anova(R2_Nagelkerke_sites, fractions = fractions)
  
  # precalculates reduced ANOVA tables
  calculateResults <- function(res, anova_rows) {
    rownames(res) = res$models
    res = res[anova_rows,]
    res$models = names(anova_rows)
    res = res[-1,c(1, 4, 3,5,6)]
    rownames(res) = res$models
    res = res[,-1]
    return(res)
  }
  
  if(inherits(object, "spatial")) {
    printFull = results[[1]][1:8,c(1, 4, 3,5,6)]
    rownames(printFull) = printFull$models
    printFull = printFull[,-1]
    rownames(printFull) = c("Abiotic", "Associations","Spatial", "Shared Abiotic+Associations", "Shared Abiotic+Spatial", "Shared Spatial+Associations", "Shared Abiotic+Associations+Spatial", "Full")
    
    anova_rows = c("Null", "F_A", "F_B", "F_S", "Full")
    names(anova_rows) = c("Null", "Abiotic", "Associations", "Spatial", "Full")
    
    toPrint = list(all = printFull)
    toPrint$discard = calculateResults(results[[1]], anova_rows)
    toPrint$proportional = calculateResults(results[[2]], anova_rows)
    toPrint$equal = calculateResults(results[[3]], anova_rows)
    
    anova_rows = c("Null", "F_A", "F_B", "F_S","F_AS", "Full")
    names(anova_rows) = c("Null", "Abiotic", "Associations", "Spatial", "Shared Abiotic+Spatial", "Full")   
    toPrint$mvp = calculateResults(results[[4]], anova_rows)
    anova_rows = c("Null", "F_A", "F_B", "F_S","Full")
    names(anova_rows) = c("Null", "Abiotic", "Associations", "Spatial", "Full")   
    toPrint$mvp_proportional = calculateResults(results[[5]], anova_rows)
    
  } else {
    
    anova_rows = c("Null", "F_A", "F_B", "Full")
    names(anova_rows) = c("Null", "Abiotic", "Associations", "Full")
    
    printFull = results[[1]][1:4,c(1, 4, 3,5,6)]
    rownames(printFull) = printFull$models
    printFull = printFull[,-1]
    rownames(printFull) = c("Abiotic", "Associations", "Shared Abiotic+Associations", "Full")
    
    toPrint = list(all = printFull)
    toPrint$discard = calculateResults(results[[1]], anova_rows)
    toPrint$proportional = calculateResults(results[[2]], anova_rows)
    toPrint$equal = calculateResults(results[[3]], anova_rows)
    toPrint$mvp = calculateResults(results[[4]], anova_rows)
    toPrint$mvp_proportional = calculateResults(results[[5]], anova_rows)
  }
  
  
  
  
  out$results = results[[1]] # TODO Max: check - das hier sind doch die einzigen Resultate die wir brauchen, oder? Es gibt doch eigentlich nur eine Aufteilung
  out$to_print = toPrint
  out$N = nrow(object$data$Y)
  out$species = list(Residual_deviance = Residual_deviance_ind,
                     Deviance = Deviance_ind,
                     R2_Nagelkerke = R2_Nagelkerke_ind,
                     R2_McFadden = R2_McFadden_ind,
                     R2_Nagelkerke_shared = R2_Nagelkerke_ind_shared,
                     R2_McFadden_shared = R2_McFadden_ind_shared                
  )
  out$sites = list(R2_Nagelkerke = R2_Nagelkerke_sites,
                   R2_McFadden = R2_McFadden_sites,
                   R2_Nagelkerke_shared = R2_Nagelkerke_sites_shared,
                   R2_McFadden_shared = R2_McFadden_sites_shared)
  out$lls = list(results_ind)
  out$proposal = proposal
  out$object = object
  class(out) = "sjSDManova"
  return(out)
}

# TODO - scale instead of 
correct_R2 = function(R2) {
  R2 = ifelse(R2 < 0, 0, R2)
  R2 = ifelse(R2 > 1.000, 0, R2)
  return(R2)
}

# Negative log-likelihood of the saturated model, which `~as.factor(1:n)` was fitted to
# approximate until 1.1.0. One free mean per site x species makes the likelihood factorise
# over cells, so mu = Y and the maximum is closed form. theta is held at the fitted value
# rather than maximised with it: the supremum over theta is degenerate (normal diverges as
# sd -> 0, nbinom only reaches the Poisson value as r -> Inf), and a deviance conditional on
# the fitted dispersion is the GLM convention.
#
# SAT_m cancels from Deviance (= RD[Null] - RD) and appears in neither R2, so it feeds only
# the `Residual deviance` column -- and the SGD fit got that column wrong: on a 100 x 8
# binomial example the refit returned -204.2 where the answer is 0, a value set by `iter`,
# `learning_rate` and `weight_decay` of a nuisance model. Dropping the fit removes the
# single largest phase of anova(), 31 % at 775 x 201.
saturated_nll = function(Y, link, theta = NULL) {
  theta_m = function() matrix(theta, nrow(Y), ncol(Y), byrow = TRUE)
  ll = switch(link,
              probit = ,
              logit  = ,
              linear = matrix(0.0, nrow(Y), ncol(Y)),
              count  = Y * log(pmax(Y, 1)) - Y - lgamma(Y + 1.0),
              nbinom = {
                r = 1.0 / (softplus(theta_m()) + 0.0001)
                p = pmin(pmax((1.0 - r / (r + Y)) + 0.0001, 0.0), 1.0 - 0.0001)
                lgamma(r + Y) - lgamma(Y + 1.0) - lgamma(r) + r * log1p(-p) + Y * log(p)
              },
              normal = -theta_m() - 0.5 * log(2 * pi))
  ll[is.na(Y)] = 0.0
  -ll
}

# The refits keep the fitted model's covariance settings. A fresh bioticStruct() would drop
# df, lambda and alpha, so a tuned model was differenced against an unpenalised sub-model of a
# different rank. A sub-model without associations has no latent term at all (`zero`).
anova_biotic = function(object, diag) {
  b = object$settings$biotic
  b$zero = diag
  b
}

# The effective sample size is a property of each sub-model's integration, not of the anova as a
# whole: the association-only model leans entirely on the latent factors and is always the worst
# integrated of the set. One column per sub-model, one row per site.
collect_ess = function(models) {
  e = vapply(models, function(x) attr(x, "ess"), numeric(nrow(models[[1]])))
  colnames(e) = names(models)
  attr(e, "newton_decrement") =
    vapply(models, function(x) { g = attr(x, "maxdec"); if (is.null(g)) NA_real_ else g }, 1.0)
  e
}

# The random intercepts of the fitted bars, one per grouping factor, as the base formula of the
# null and of every sub-model; ~1 for a model without bars.
anova_ri_formula = function(object) {
  re = Filter(function(b) isTRUE(b$intercept), object$settings$env$re)
  re = re[!duplicated(vapply(re, function(b) b$group, ""))]
  ri = vapply(re, function(b)
    if (b$loading == "shared") sprintf("re(1 | %s, loading = \"shared\")", b$group)
    else sprintf("re(1 | %s, df = %dL)", b$group, b$df), "")
  stats::as.formula(paste("~", paste(c("1", ri), collapse = " + ")),
                    env = environment(object$settings$env$formula))
}

# sites coupled through a shared group of any bar; the marginal likelihood factorises over them
re_components = function(index) {
  comp = seq_along(index[[1]])
  repeat {
    old = comp
    for (g in index) comp = pmin(comp, stats::ave(comp, g, FUN = min))
    if (identical(old, comp)) break
  }
  match(comp, unique(comp))
}

# Marginal log-likelihood of a model with random blocks. The group effects u are integrated out
# by importance sampling from their variational posterior q (weights p(u)/q(u), `re_draws`
# joint draws), the latent factors per site by the usual estimator at every draw, with one set of
# inner draws shared across the outer ones. A plug-in at the posterior means is a conditional
# likelihood and would not be comparable with the sub-models. Per-site values are the
# posterior-weighted conditional log-likelihoods plus an equal share of their component's
# integration term, so they sum to the component's marginal (a convention).
re_marginal = function(m, sampling = 1000L, proposal = "prior", re_draws = 50L, mc_batch = NULL, ...) {
  model = m$model
  net = model$net
  net$eval()
  d = model$as_tensors(c(as_input(config_X(m$settings$env)), as_input(config_X(m$spatial))),
                       m$data$Y)
  inp = d$inputs
  Y = d$Y
  n = Y$shape[1]
  S = Y$shape[2]
  zero = isTRUE(m$settings$biotic$zero)
  sampling = if (zero) 1L else as.integer(sampling)
  draws = as.integer(re_draws)
  dev = model$device
  dt = model$dtype
  sigma = model$sigma$detach()
  theta = model$theta
  link = m$family$link
  derivs = if (proposal == "laplace" && !zero) sjsdm_derivs(link, model$alpha) else NULL
  K = length(net$re_at)
  blocks = lapply(seq_len(K), function(k) net$random[[k]])
  index = lapply(seq_len(K), function(k)
    as.integer(inp[[net$re_at[k]]]$select(2, blocks[[k]]$q + 1L)$cpu()))
  comp = re_components(index)
  C = max(comp)

  torch::with_no_grad({
    mu_fixed = net(inp, FALSE)
    for (k in seq_len(K)) mu_fixed = mu_fixed$sub(blocks[[k]](inp[[net$re_at[k]]], FALSE))
    # outer draws per bar and group, and their log p(u) - log q(u) summed per component
    eps = lapply(blocks, function(b) torch::torch_randn(c(draws, b$nlevels, b$d), dtype = dt, device = dev))
    W = matrix(0, draws, C)
    for (k in seq_len(K)) {
      b = blocks[[k]]
      L = b$L_mat()
      u = b$m$unsqueeze(1)$add(torch::torch_matmul(L$unsqueeze(1), eps[[k]]$unsqueeze(4))$squeeze(4))
      logdet = as.numeric(torch::torch_diagonal(L, dim1 = 2, dim2 = 3)$log()$sum(dim = 2)$cpu())
      lw = as.matrix(eps[[k]]$pow(2)$sum(dim = 3)$sub(u$pow(2)$sum(dim = 3))$mul(0.5)$cpu())
      lw = sweep(lw, 2, logdet, "+")
      gc_ = vapply(seq_len(b$nlevels), function(g) comp[match(g, index[[k]])], 1L)
      for (g in which(!is.na(gc_))) W[, gc_[g]] = W[, gc_[g]] + lw[, g]
    }
  })

  if (is.null(mc_batch))
    mc_batch = min(as.integer(m$settings$step_size), as.integer(2.5e9 / (24 * as.numeric(sampling) * S)))
  mc_batch = max(1L, as.integer(mc_batch))
  Lk = matrix(0, draws, n)
  cond = array(0, c(draws, n, S))
  cell = if (zero) array(0, c(draws, n, S)) else NULL
  maxdec = if (!is.null(derivs)) 0 else NA_real_
  for (start in seq(1L, n, by = mc_batch)) {
    rows = start:min(start + mc_batch - 1L, n)
    r1 = torch::torch_tensor(rows, dtype = torch::torch_long(), device = dev)
    Y_b = Y$index_select(1, r1)
    inp_b = lapply(inp, function(z) if (is.null(z)) NULL else z$index_select(1, r1))
    torch::with_no_grad({
      mu_b = mu_fixed$index_select(1, r1)
      if (zero) {
        noise = torch::torch_zeros(c(1L, length(rows), sigma$shape[2]), dtype = dt, device = dev)
        lw_in = NULL
      } else if (!is.null(derivs)) {
        mean_b = mu_b
        for (k in seq_len(K)) mean_b = mean_b$add(blocks[[k]](inp_b[[net$re_at[k]]], FALSE))
        prop = mvp_laplace_proposal(mean_b, Y_b, sigma, derivs, theta)
        maxdec = max(maxdec, prop$maxdec)
        dr = mvp_laplace_draw(prop, sampling)
        noise = dr$z
        lw_in = dr$lw
      } else {
        noise = torch::torch_randn(c(sampling, length(rows), sigma$shape[2]), dtype = dt, device = dev)
        lw_in = NULL
      }
      for (j in seq_len(draws)) {
        mu_j = mu_b
        for (k in seq_len(K)) {
          x = inp_b[[net$re_at[k]]]
          gi = x$select(2, blocks[[k]]$q + 1L)$clamp(min = 1)$to(dtype = torch::torch_long())
          mu_j = mu_j$add(blocks[[k]](x, TRUE, eps[[k]][j, , ]$index_select(1, gi)))
        }
        lp = mvp_species_logprob(mu_j, Y_b, sigma, link = link, alpha = model$alpha,
                                 sampling = sampling, theta = theta, noise = noise)
        jt = lp$sum(dim = 3)
        if (!is.null(lw_in)) jt = jt$add(lw_in)
        joint = mvp_mc_nll(jt)
        cf = lp$cumsum(dim = 3)
        cb = lp$flip(3)$cumsum(dim = 3)$flip(3)
        z0 = torch::torch_zeros_like(cf$select(3, 1)$unsqueeze(3))
        loo = torch::torch_cat(list(z0, cf$narrow(3, 1, S - 1L)), dim = 3)$
          add(torch::torch_cat(list(cb$narrow(3, 2, S - 1L), z0), dim = 3))
        if (!is.null(lw_in)) loo = loo$add(lw_in$unsqueeze(3))
        Lk[j, rows] = -as.numeric(joint$cpu())
        cond[j, rows, ] = as.matrix(joint$unsqueeze(2)$sub(mvp_mc_nll(loo))$cpu())
        if (zero) cell[j, rows, ] = as.matrix(lp$select(1, 1)$cpu())
      }
    })
  }

  # per component: total log marginal and posterior weights over the outer draws
  joint = numeric(n)
  ess = numeric(n)
  pi_ = matrix(0, draws, n)
  for (c in seq_len(C)) {
    i = which(comp == c)
    a = W[, c] + rowSums(Lk[, i, drop = FALSE])
    tot = max(a) + log(mean(exp(a - max(a))))
    w = exp(a - max(a))
    w = w / sum(w)
    pi_[, i] = w
    e = drop(w %*% Lk[, i, drop = FALSE])
    joint[i] = e + (tot - sum(e)) / length(i)
    ess[i] = 1 / sum(w^2) / draws
  }
  wsum = function(A) apply(A * as.vector(pi_), c(2, 3), sum)
  out = list(joint = -joint, conditional = wsum(cond), ess = ess, maxdec = maxdec)
  if (zero) {
    cl = wsum(cell)
    obs = !is.na(m$data$Y)
    shift = (joint - rowSums(cl)) / pmax(rowSums(obs), 1)
    out$cell = cl + shift * obs
  }
  out
}

get_conditional_lls = function(m, null_m, mc_batch = NULL, proposal = "prior", ...) {
  if (length(m$settings$env$re)) {
    r = re_marginal(m, mc_batch = mc_batch, proposal = proposal, ...)
    diff_ll = colSums(null_m - r$conditional)
    rates = diff_ll / sum(diff_ll)
    out = null_m - matrix(rates, nrow(null_m), ncol(null_m), byrow = TRUE) * (rowSums(null_m) - r$joint)
    attr(out, "ess") = r$ess
    attr(out, "maxdec") = r$maxdec
    return(out)
  }
  
  predictions = predict(m, type = "raw")  
  args = list(...)
  if(!is.null(args[["sampling"]])) {
    samples = args[["sampling"]]
  } else {
    samples = m$settings$sampling
  }
  
  # one shared set of MC samples, so the joint and the leave-one-out likelihoods
  # are compared under the same noise
  sigma = m$model$sigma$detach()
  tt = function(x) sjsdm_tensor(x, m$model$dtype, m$model$device)
  theta = m$model$theta
  derivs = if (proposal == "laplace") sjsdm_derivs(m$family$link, m$model$alpha) else NULL
  if (proposal == "laplace" && is.null(derivs)) {
    message("link '", m$family$link, "' has no curvature for the Laplace proposal, ",
            "falling back to prior sampling")
    proposal = "prior"
  }
  MC_samples = if (proposal == "prior")
    torch::torch_randn(c(as.integer(samples), nrow(predictions), ncol(sigma)),
                       dtype = m$model$dtype, device = m$model$device)

  n = nrow(predictions)
  S = ncol(m$data$Y)

  # Row batches. The default stays at step_size because the batched matmul is not bitwise
  # invariant to the row grouping (one float32 ulp on eta), so changing it silently moves
  # every anova that has already been run. Memory only ever lowers it: about five
  # [samples, rows, species] float32 tensors are alive at the peak, capped here at ~2.5 GB.
  if (is.null(mc_batch))
    mc_batch = min(as.integer(m$settings$step_size),
                   as.integer(2.5e9 / (24 * as.numeric(samples) * S)))
  mc_batch = max(1L, as.integer(mc_batch))

  joint_ll = numeric(n)
  raw_ll = matrix(0, n, S)
  ess = numeric(n)
  maxdec = if (proposal == "laplace") 0 else NA_real_
  for (start in seq(1L, n, by = mc_batch)) {
    k = start:min(start + mc_batch - 1L, n)
    mu_k = tt(predictions[k, , drop = FALSE])
    Y_k = tt(m$data$Y[k, , drop = FALSE])
    if (proposal == "laplace") {
      prop = mvp_laplace_proposal(mu_k, Y_k, sigma, derivs, theta)
      maxdec = max(maxdec, prop$maxdec)
      dr = mvp_laplace_draw(prop, as.integer(samples))
      noise_k = dr$z
      lw = dr$lw
    } else {
      noise_k = MC_samples[, k, , drop = FALSE]
      lw = NULL
    }
    lp = mvp_species_logprob(mu_k, Y_k, sigma,
                             link = m$family$link, alpha = m$model$alpha,
                             sampling = as.integer(samples), theta = theta,
                             noise = noise_k)
    # Every link is elementwise in the species index, so column j of lp is what a refit
    # without the other species would have produced. All S leave-one-out sums come from a
    # forward and a backward cumulative sum: LOO_j = sum_{k<j} + sum_{k>j}. Subtracting
    # column j from the total instead is one kernel cheaper but cancels catastrophically --
    # a species log-probability of order 1 against a sum over S of order S -- and moved the
    # result by 0.1% at S = 150. Slicing S-1 columns per species is exact but costs S times
    # as much.
    cf = lp$cumsum(dim = 3)
    cb = lp$flip(3)$cumsum(dim = 3)$flip(3)
    # the joint stays a plain sum reduction: cf's last column is the same quantity but
    # accumulated in a different order, and it feeds every fraction
    # the importance weight is constant in the species index, so it enters the joint and every
    # leave-one-out sum the same way and the cumsum trick is untouched
    jt = lp$sum(dim = 3)
    if (!is.null(lw)) jt = jt$add(lw)
    joint_ll[k] = as.numeric(mvp_mc_nll(jt)$cpu())
    ess[k] = as.numeric(mvp_ess(jt)$cpu())
    z = torch::torch_zeros_like(cf$select(3, 1)$unsqueeze(3))
    left = torch::torch_cat(list(z, cf$narrow(3, 1, S - 1L)), dim = 3)
    right = torch::torch_cat(list(cb$narrow(3, 2, S - 1L), z), dim = 3)
    loo = left$add(right)
    if (!is.null(lw)) loo = loo$add(lw$unsqueeze(3))
    raw_ll[k, ] = as.matrix(mvp_mc_nll(loo)$cpu())
  }

  raw_conditional_ll = -( (-joint_ll) - (-raw_ll ))
  diff_ll = colSums(null_m - raw_conditional_ll)
  rates = diff_ll/sum(diff_ll)
  rescaled_conditional_lls = null_m - matrix(rates, nrow = nrow(m$data$Y), ncol = ncol(m$data$Y), byrow = TRUE) * (rowSums(null_m)-joint_ll)
  # A site whose mode was missed gets a proposal centred away from the mass, and the median ESS
  # does not show it because a failing minority does not move a median. A decrement of 1e-2
  # bounds the log-density error at the mode by about 5e-5.
  if (isTRUE(maxdec > 1e-2))
    warning("Laplace proposal: the mode was not reached for at least one site (Newton decrement ",
            signif(maxdec, 3), "). Check the per-site `ess` and compare against ",
            "proposal = \"prior\".", call. = FALSE)
  attr(rescaled_conditional_lls, "ess") = ess / as.integer(samples)
  attr(rescaled_conditional_lls, "maxdec") = maxdec
  
  return(rescaled_conditional_lls)
}

get_shared_anova = function(R2objt, fractions = c("2", "3")) {
  fractions = match.arg(fractions)
  if(fractions == "3") {
    F_A <- R2objt$Full - R2objt$BS
    F_B <- R2objt$Full - R2objt$AS
    F_S <- R2objt$Full - R2objt$AB
    F_BS <- R2objt$Full - R2objt$A - (F_B + F_S)
    F_AB <- R2objt$Full - R2objt$S - (F_A + F_B)
    F_AS <- R2objt$Full - R2objt$B - (F_A + F_S)
    F_ABS <- R2objt$Full - (F_A + F_B + F_S + F_BS + F_AB + F_AS)
    
    A = F_A + F_AB/2 + F_AS/2 + F_ABS/3
    B = F_B + F_AB/2 + F_BS/2 + F_ABS/3
    S = F_S + F_AS/2 + F_BS/2 + F_ABS/3
    equal = list(F_A = A, F_B = B, F_S = S, R2 = R2objt$Full)
    
    FF_A = R2objt$AS - R2objt$S
    FF_S = R2objt$AS - R2objt$A
    FF_AS = R2objt$AS - FF_A - FF_S
    FF_B = R2objt$Full - R2objt$AS
    
    
    mvp = list(F_A = FF_A, F_B = FF_B, F_S = FF_S, R2 = R2objt$Full)
    
    mvp_proportional = list(F_A = FF_A+FF_AS*abs(FF_A)/(abs(FF_A)+abs(FF_S)), 
                            F_B = FF_B, 
                            F_S = FF_S+FF_AS*abs(FF_S)/(abs(FF_A)+abs(FF_S)), 
                            R2 = R2objt$Full)
    
    # 3F on 2F:
    
    m_f = function(x) abs(x)
    
    shared_total = (m_f(F_AB)+m_f(F_BS)+m_f(F_AS))
    F_AB = F_AB + F_ABS*m_f(F_AB)/shared_total
    F_BS = F_BS + F_ABS*m_f(F_BS)/shared_total
    F_AS = F_AS + F_ABS*m_f(F_AS)/shared_total
    
    # 2F on 1F:
    A = F_A + F_AB*m_f(F_A)/(m_f(F_A)+m_f(F_B)) + F_AS*m_f(F_A)/(m_f(F_S)+m_f(F_A))
    
    B = F_B + F_AB*m_f(F_B)/(m_f(F_A)+m_f(F_B)) + F_BS*m_f(F_B)/(m_f(F_S)+m_f(F_B))
    
    S = F_S + F_AS*m_f(F_S)/(m_f(F_S)+m_f(F_A)) + F_BS*m_f(F_S)/(m_f(F_S)+m_f(F_B))
    
    # 
    # 
    # A = F_A + F_AB*abs(F_A)/(abs(F_A)+abs(F_B)) + F_AS*abs(F_A)/(abs(F_S)+abs(F_A)) + F_ABS*abs(F_A)/(abs(F_A)+abs(F_B)+abs(F_S))
    # B = F_B + F_AB*abs(F_B)/(abs(F_A)+abs(F_B)) + F_BS*abs(F_B)/(abs(F_S)+abs(F_B)) + F_ABS*abs(F_B)/(abs(F_A)+abs(F_B)+abs(F_S))
    # S = F_S + F_AS*abs(F_S)/(abs(F_S)+abs(F_A)) + F_BS*abs(F_S)/(abs(F_S)+abs(F_B)) + F_ABS*abs(F_S)/(abs(F_A)+abs(F_B)+abs(F_S))
    # R2 = correct_R2(R2objt$Full) TODO Check that this can be gone
    proportional = list(F_A = A, F_B = B, F_S = S, R2 = R2objt$Full)
    
  } else {
    F_A = R2objt$Full-R2objt$B
    F_B =  R2objt$Full-R2objt$A
    F_AB = R2objt$Full - F_A -F_B
    A = F_A + F_AB*abs(F_A)/(abs(F_A)+abs(F_B))
    B = F_B + F_AB*abs(F_B)/(abs(F_A)+abs(F_B))
    S = 0
    
    mvp = list(F_A = R2objt$A, F_B = R2objt$Full - R2objt$A, F_S = S, R2 = R2objt$Full)
    mvp_shared = mvp
    mvp_proportional = mvp_shared
    proportional = list(F_A = A, F_B = B, F_S = S, R2 = R2objt$Full)
    A = F_A + F_AB*0.5
    B = F_B + F_AB*0.5
    S = 0
    equal = list(F_A = A, F_B = B, F_S = S, R2 = R2objt$Full)
  }
  return(list(proportional = proportional, equal = equal, mvp = mvp, mvp_proportional = mvp_proportional))
}

get_null_ll = function(object, verbose = TRUE, condition = FALSE, ...) {
  
  object_tmp = object
  object_tmp$settings$se = FALSE
  base = if(condition) anova_ri_formula(object) else ~1
  
  if(inherits(object, "spatial")) {
    null_model = update(object_tmp, env_formula = base, spatial_formula = ~0, biotic = anova_biotic(object, TRUE), verbose = verbose)
  } else {
    null_model = update(object_tmp, env_formula = base, biotic = anova_biotic(object, TRUE), verbose = verbose)
  }
  # with random intercepts the null has no closed form: integrate them out
  if(length(null_model$settings$env$re)) return(re_marginal(null_model, ...)$cell)
  null_pred = predict(null_model)
  
  if(object$family$family$family == "binomial") {
    null_m = stats::dbinom( object$data$Y, 1, null_pred, log = TRUE)
  } else if(object$family$family$family == "poisson") {
    null_m = stats::dpois( object$data$Y, null_pred, log = TRUE)
  } else if(object$family$family$family == "nbinom") {
    check_module()
    theta = null_model$theta
    theta = 1.0/(softplus(theta)+0.0001)
    theta = matrix(theta, nrow = nrow(null_pred), ncol = ncol(null_pred), byrow = TRUE)
    probs = (1.0 - theta/(theta+null_pred))+0.0001
    probs = ifelse(probs < 0.0, 0.0, probs)
    probs = ifelse(probs > 1.0, 1.0-0.0001, probs )
    Y = object$data$Y
    null_m = lgamma(theta + Y) - lgamma(Y + 1) - lgamma(theta) +
      theta * log1p(-probs) + Y * log(probs)
  } else if(object$family$family$family == "gaussian") {
    #warning("family = gaussian() is not fully supported yet.")
    sd_marginal = sqrt(rowSums(null_model$model$get_sigma^2) + exp(2.0 * null_model$theta))
    null_m = sapply(1:ncol(object$data$Y), function(i) stats::dnorm(object$data$Y[,i], null_pred[,i], sd = sd_marginal[i], log = TRUE))
  }
  # missing responses contribute nothing, as in the fitted and saturated likelihoods
  null_m[is.na(object$data$Y)] = 0
  return(null_m)
}



#' Summary table of sjSDM anova
#' 
#' The function prints and returns invisible a summary table of an sjSDM ANOVA, created by \code{\link{anova.sjSDM}}
#' 
#' @param object an object of \code{\link{anova.sjSDM}}
#' @param method method used to calculate the ANOVA 
#' @param fractions how to handle the shared fractions. See details
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' 
#' @details The function returns a ANOVA table with Deviance as well as the pseudo-R2 metrics of Nagelkerke and McFadden
#' 
#' There are six options to handle shared ANOVA fractions, which is variance that can be explained, typically as a result of collinearity, by several of the fractions:
#' 
#' 1. "mvp_proportional" (the default) splits Abiotic and Spatial within the model without associations, distributes their shared part proportionally, and gives the remainder of the full model to the associations. Without a spatial component it is the same as "proportional"
#' 2. "mvp" is the same split without distributing the shared Abiotic and Spatial part. Without a spatial component it is the same as "proportional"
#' 3. "all" returns the shared fractions explicitly
#' 4. "discard" discards the fractions, as typically in a type II Anova
#' 5. "proportional" distributes shared fractions proportional to the unique fractions, weighted by their absolute size, so a negative unique fraction still attracts shared variance
#' 6. "equal" distributes shared fractions equally to the unique fractions
#' 
#' @return The matrix that is printed out is silently returned
#' 
#' @example /inst/examples/anova-example.R
#' @export
summary.sjSDManova = function(object, 
                              method = c("ANOVA"),
                              fractions = c("mvp_proportional", "mvp", "all","discard", "proportional", "equal"), ...) {
  cat("Analysis of Deviance Table\n\n")
  method = match.arg(method)
  fractions = match.arg(fractions)
  
  out = object$to_print[[fractions]]
  stats::printCoefmat(out)
  if(!is.null(object$grouping))
    cat(sprintf("\nConditional on the random intercepts, which alone explain R2 McFadden %.4f\n",
                object$grouping[["R2_McFadden"]]))
  return(invisible(out))
  
}


#' Print sjSDM anova object
#' 
#' This is a wrapper for \code{\link{summary.sjSDManova}}, maintained for backwards compatibility - prefer to use summary() instead
#'
#' @param x an object of type sjSDManova created by \code{\link{anova.sjSDM}}
#' @param ... additional arguments to \code{\link{summary.sjSDManova}}
#' @return The table printed by \code{\link{summary.sjSDManova}}, invisibly.
#' 
#' @example /inst/examples/anova-example.R
#' @export
print.sjSDManova = function(x,...) {
  out = summary(x, ...)
  return(invisible(out))
}


#' Plot anova results
#' 
#' 
#' @param x anova object from \code{\link{anova.sjSDM}}
#' @param y unused argument
#' @param type deviance, Nagelkerke or McFadden R-squared
#' @param fractions how to handle shared fractions
#' @param cols colors for the groups
#' @param alpha alpha for colors
#' @param env_deviance environmental deviance
#' @param ... no function, for compatibility with the generic
#' 
#' 
#' @return 
#' 
#' List with the following components:
#' 
#' \item{VENN}{Matrix of shown results.}
#' 
#' @references 
#' Leibold, M. A., Rudolph, F. J., Blanchet, F. G., De Meester, L., Gravel, D., Hartig, F., ... & Chase, J. M. (2022). The internal structure of metacommunities. Oikos, 2022(1).
#' 
#' @export
plot.sjSDManova = function(x, 
                           y, 
                           type = c( "McFadden", "Deviance", "Nagelkerke"), 
                           fractions = c("discard", "proportional", "equal"),
                           cols = c("#7FC97F","#BEAED4","#FDC086"),
                           alpha=0.15, 
                           env_deviance = NULL,
                           ...) {
  
  fractions = match.arg(fractions)
  
  lineSeq = 0.3
  nseg = 100
  dr = 1.0
  type = match.arg(type)
  out = list()
  
  oldpar = par(no.readonly = TRUE)
  on.exit(par(oldpar))
  
  values = x$results
  
  select_rows = 
    if(x$spatial) { 
      sapply(c("F_A", "F_B", "F_AB","F_S", "F_AS", "F_BS", "F_ABS"), function(i) which(values$models == i, arr.ind = TRUE))
    } else {
      sapply(c("F_A", "F_B", "F_AB"), function(i) which(values$models == i, arr.ind = TRUE))
    }
  
  values = values[select_rows,]
  col_index = 
    switch (type,
            Deviance = 4,
            Nagelkerke = 5,
            McFadden = 6
    )
  
  
  graphics::plot(NULL, NULL, xlim = c(0,1), ylim =c(0,1),pty="s", axes = FALSE, xlab = "", ylab = "")
  xx = 1.1*lineSeq*cos( seq(0,2*pi, length.out=nseg))
  yy = 1.1*lineSeq*sin( seq(0,2*pi, length.out=nseg))
  graphics::polygon(xx+lineSeq,yy+(1-lineSeq), col= addA(cols[1],alpha = alpha), border = "black", lty = 1, lwd = 1)
  graphics::text(lineSeq-0.1, (1-lineSeq),labels = round(values[1,col_index],3))
  graphics::text(mean(xx+lineSeq), 0.9,labels = "Environmental", pos = 3)
  
  graphics::polygon(xx+1-lineSeq,yy+1-lineSeq, col= addA(cols[2],alpha = alpha), border = "black", lty = 1, lwd = 1)
  graphics::text(1-lineSeq+0.1, (1-lineSeq),labels = round(values[2,col_index],3))
  graphics::text(1-mean(xx+lineSeq), 0.9,labels = "Associations", pos = 3)
  graphics::text(0.5, (1-lineSeq),labels = round(values[3,col_index],3))
  
  if(x$spatial) {
    graphics::polygon(xx+0.5,yy+lineSeq, col= addA(cols[3],alpha = alpha), border = "black", lty = 1, lwd = 1)
    graphics::text(0.5, lineSeq+0.0,pos = 1,labels = round(values[4,col_index],3))
    graphics::text(0.5, 0.1,labels = "Spatial", pos = 1)
    graphics::text(0.3, 0.5,pos=1,labels   = round(values[5,col_index],3)) # AS
    graphics::text(1-0.3, 0.5,pos=1,labels = round(values[6,col_index],3)) # BS
    graphics::text(0.5, 0.5+0.05,labels    = round(values[7,col_index],3)) # ABS
  }
  out$VENN = values
  return(invisible(out))
}





