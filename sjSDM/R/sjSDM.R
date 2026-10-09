#' @title Fitting scalable joint Species Distribution Models (sjSDM)
#'
#' @description 
#' 
#' \code{sjSDM} is used to fit joint Species Distribution models (jSDMs) using the central processing unit (CPU) or the graphical processing unit (GPU). 
#' The default is a multivariate probit model based on a Monte-Carlo approximation of the joint likelihood. 
#' \code{sjSDM} can be used to fit linear but also deep neural networks and supports the well known formula syntax. 
#' The environmental formula carries structure: it may hold \code{\link{NN}} blocks and 'lme4' random-effect bars
#' beside the parametric terms, see the details.
#' 
#' @param Y matrix of species occurrences/responses in range
#' @param env matrix of environmental predictors, object of type \code{\link{linear}} or \code{\link{DNN}}. A
#'   \code{\link{linear}} formula may contain \code{\link{NN}} blocks and random-effect bars, see details
#' @param biotic defines biotic (species-species associations) structure, object of type \code{\link{bioticStruct}}
#' @param spatial defines spatial structure, object of type \code{\link{linear}} or \code{\link{DNN}}
#' @param family error distribution with link function, see details for supported distributions
#' @param iter number of fitting iterations
#' @param step_size batch size for stochastic gradient descent, if \code{NULL} then step_size is set to: \code{step_size = 0.1*nrow(X)}
#' @param learning_rate learning rate for the optimizer, see \code{\link{sjSDMControl}}
#' @param se calculate standard errors for environmental coefficients
#' @param sampling number of sampling steps for Monte Carlo integration
#' @param parallel number of cpu cores for the data loader, only necessary for large datasets
#' @param control control parameters for optimizer, see \code{\link{sjSDMControl}}
#' @param device which device to be used, \code{"cpu"}, \code{"gpu"}, \code{"mps"}, or the integer index of a
#'   CUDA device. A GPU request falls back to the CPU with a warning when no CUDA device is present
#' @param dtype which data type, most GPUs support only 32 bit floats.
#' @param seed seed for random operations
#' @param verbose `TRUE` or `FALSE`, indicating whether progress should be printed or not
#' 
#' @details 
#' \loadmathjax
#' The function fits per default a multivariate probit model via Monte-Carlo integration (see Chen et al., 2018) of the joint likelihood for all species.
#' 
#' \subsection{Model description}{
#' 
#' The most common jSDM structure describes the site (\mjseqn{i = 1, ..., I})  by species (\mjseqn{j = 1, ..., J}) matrix \mjseqn{Y_{ij}} as a function of
#' environmental covariates \mjseqn{X_{in}}(\mjseqn{n=1,...,N} covariates), and the species-species covariance matrix
#' \mjseqn{\Sigma} accounts for correlations in \mjseqn{e_{ij}}:
#' 
#' \mjsdeqn{g(Z_{ij}) = \beta_{j0} + \Sigma^{N}_{n=1}X_{in}\beta_{nj} + e_{ij}}
#' 
#' with \mjseqn{g(.)} as link function. For the multivariate probit model, the link function is:
#' 
#' \mjsdeqn{Y_{ij}=1(Z_{ij} > 0)}
#' 
#' The probability to observe the occurrence vector \mjseqn{\bf{Y_i}} is:
#' 
#' \mjsdeqn{Pr(\bf{Y}_i|\bf{X}_i\beta, \Sigma) = \int_{A_{iJ}}...\int_{A_{i1}} \phi_J(\bf{Y}_i^{\ast};\bf{X}_i\beta, \Sigma)  dY_{i1}^{\ast}... dY_{iJ}^{\ast}}
#' 
#' in the interval \mjseqn{A_{ij}} with \mjseqn{(-\inf, 0]} if \mjseqn{Y_{ij}=0} and \mjseqn{ [0, +\inf) }  if \mjseqn{Y_{ij}=1}.
#' 
#' and \mjseqn{\phi} being the density function of the multivariate normal distribution. 
#' 
#' The probability of \mjseqn{\bf{Y_i}} requires to integrate over \mjseqn{\bf{Y_i^{\ast}}} which has no closed analytical expression for more than two species
#' which makes the evaluation of the likelihood computationally costly and needs a numerical approximation.
#' The previous equation can be expressed more generally as:
#' 
#' \mjsdeqn{ \mathcal{L}(\beta, \Sigma; \bf{Y}_i, \bf{X}_i) = \int_{\Omega} \prod_{j=1}^J Pr(Y_{ij}|\bf{X}_i\beta+\zeta) Pr(\zeta|\Sigma) d\zeta  }
#' 
#' \code{sjSDM} approximates this integral by \mjseqn{M} Monte-Carlo samples from the multivariate normal species-species covariance. 
#' After integrating out the covariance term, the remaining part of the likelihood can be calculated as in an univariate case and the average 
#' of the \mjseqn{M} samples are used to get an approximation of the integral:
#' 
#' \mjsdeqn{ \mathcal{L}(\beta, \Sigma; \bf{Y}_i, \bf{X}_i) \approx \frac{1}{M} \Sigma_{m=1}^M \prod_{j=1}^J Pr(Y_{ij}|\bf{X}_i\beta+\zeta_m)}
#' 
#' with \mjseqn{ \zeta_m \sim MVN(0, \Sigma)}. 
#' 
#' \code{sjSDM} uses 'torch' (libtorch) and can optionally run the model on the graphical
#' processing unit (GPU), see \code{\link{installation_help}} for how to enable that.
#' 
#' See Pichler and Hartig, 2020 for benchmark results.
#' }
#' 
#' \subsection{Supported distributions}{
#' 
#' Currently supported distributions and link functions, which are :
#' \itemize{
#' \item \code{\link{binomial}}: \code{"probit"} or \code{"logit"}
#' \item \code{\link{poisson}}: \code{"log"} 
#' \item \code{"nbinom"}: \code{"log"} 
#' \item \code{\link{gaussian}}: \code{"identity"} 
#' }
#' }
#' 
#' \subsection{Space}{
#' 
#' We can extend the model to account for spatial auto-correlation between the sites by:
#' 
#' \mjsdeqn{g(Z_{ij}) = \beta_{j0} + \Sigma^{N}_{n=1}X_{in}\beta_{nj} + \Sigma^{M}_{m=1}S_{im}\alpha_{mj} + e_{ij}}
#' 
#' There are two ways to generate spatial predictors \mjseqn{S}:
#' 
#' \itemize{
#' \item trend surface model - using spatial coordinates in a polynomial:
#' 
#'  \code{linear(data=Coords, ~0+poly(X, Y, degree = 2))} 
#'  
#' \item eigenvector spatial filtering - using spatial eigenvectors. 
#'   Spatial eigenvectors can be generated by the \code{\link{generateSpatialEV}} function:
#'   
#'   \code{SPV = generateSpatialEV(Coords)}
#'   
#'   Then we use, for example, the first 20 spatial eigenvectors:
#' 
#'   \code{linear(data=SPV[ ,1:20], ~0+.)}
#' }
#' 
#' It is important to set the intercept to 0 in the spatial term (e.g. via \code{~0+.}) because the intercept is already set in the environmental object.
#' 
#' }
#' 
#' \subsection{Structure in the environmental formula}{
#' 
#' A \code{\link{linear}} formula is not restricted to parametric terms. It may in addition hold
#' \code{\link{NN}} blocks and random-effect bars:
#' 
#' \code{linear(X, ~ temp + NN(soil + ndvi, hidden = c(50, 50)) + (temp | plot))}
#' 
#' Every block is summed on the scale of the linear predictor,
#' 
#' \mjsdeqn{g(Z_{ij}) = \bf{X}_i\beta_j + f(\bf{X}^{NN}_i)_j + \Sigma_b \Sigma_k Z^b_{ik} \bf{u}^b_{g(i)}\Lambda^b_{jk} + e_{ij}}
#' 
#' so the parametric part keeps its meaning: \code{\link{coef.sjSDM}}, \code{\link{summary.sjSDM}} and the
#' standard errors describe \mjseqn{\beta} and nothing else.
#' 
#' \code{\link{NN}(soil + ndvi)} is a feed-forward network from its own covariates to the species. It sits
#' beside the interpretable terms rather than replacing them, which is the difference to \code{\link{DNN}}.
#' The block itself has no interpretable coefficients, and the standard errors are conditional on the fitted
#' block: they ignore its uncertainty, and a warning says so. \code{\link{update.sjSDM}} with
#' \code{env_blocks} refits with a subset of the blocks, which is how a block is switched off.
#' 
#' A bar is written in 'lme4' syntax and carries 'lme4' semantics, through the 'reformulas' package:
#' \code{(1 | plot)}, \code{(temp | plot)} for a correlated intercept and slope, \code{(0 + temp | plot)},
#' \code{(temp || plot)} and \code{(1 | plot/subplot)}. \code{\link{re}} is the explicit form where a bar
#' needs arguments, \code{re(1 | plot, df = 2)}.
#' 
#' A bar is a design-level latent factor, not a scalar shift. One group effect
#' \mjseqn{\bf{u}_g \sim N(0, \bf{I}_d)} is drawn per group and loaded onto the species by
#' \mjseqn{\Lambda}, so the group effect has its own species-species covariance
#' \mjseqn{\Lambda\Lambda'} of rank \code{df} and species may respond to a plot with different magnitude
#' and sign. \code{df} defaults to \code{max(q, 5)}, with \mjseqn{q} the number of columns of the bar, and
#' must not be smaller than \mjseqn{q}: below that the intercept-slope correlation is forced to
#' \mjseqn{\pm 1}. \code{re(loading = "shared")} gives the classical mixed-model case instead, one shift per
#' group identical for every species.
#' 
#' The bars are integrated out by a stochastic variational approximation. Each group gets a Gaussian
#' posterior, drawn with the reparameterisation trick, and the Kullback-Leibler term to the
#' \mjseqn{N(0, \bf{I})} prior supplies the shrinkage; the Monte-Carlo likelihood itself is unchanged.
#' \code{\link{getRE}} reports the group effects, \code{\link{getCov}} and \code{\link{getCor}} with
#' \code{which = "re"} the covariance the bar induces.
#' }
#' 
#' \subsection{Limitations of blocks and bars}{
#' 
#' \itemize{
#'  \item \code{\link{anova.sjSDM}} refuses on a model with a bar. The random block is re-estimated in every
#'    refit, so it absorbs the fraction that was removed instead of cancelling from the difference, and the
#'    per-species rescaling is recomputed with it. Use \code{\link{getRE}} for the group variances.
#'  \item \code{\link{logLik.sjSDM}} for a model with a bar is the conditional value at the posterior means,
#'    neither the evidence lower bound nor a marginal likelihood. It is not comparable with the
#'    log-likelihood of a model fitted without the bar, and neither are information criteria built from it.
#'  \item \code{\link{sjSDM_cv}} supports neither blocks nor bars. It raises an error rather than dropping
#'    them.
#'  \item Neither is available in the spatial formula.
#'  \item The reported group variance grows with \code{df}, because the Kullback-Leibler term constrains the
#'    posterior and not the loadings. Report the rank alongside the variance, and prefer a small \code{df}.
#' }
#' }
#' 
#' \subsection{Installation}{
#' 
#' sjSDM needs the 'torch' binaries, which \code{\link{install_sjSDM}} (a wrapper around \code{torch::install_torch}) downloads. If \code{\link{sjSDM}} still does not work after restarting R, follow the trouble shooting guide \code{\link{installation_help}}.
#' If the problem remains, please create an issue on \href{https://github.com/TheoreticalEcology/s-jSDM/issues}{issue tracker} with a copy of the \code{\link{install_diagnostic}} output as a quote. 
#' }
#' 
#' @return 
#' An S3 class of type 'sjSDM' including the following components:
#' 
#' \item{cl}{Model call}
#' \item{formula}{Formula object for environmental covariates.}
#' \item{names}{Names of environmental covariates.}
#' \item{species}{Names of species (can be \code{NULL} if columns of Y are not named).}
#' \item{get_model}{Method which builds and returns the underlying 'torch' model.}
#' \item{logLik}{negative log-Likelihood of the model and the regularization loss.}
#' \item{model}{The actual model.}
#' \item{settings}{List of model settings, see arguments of \code{\link{sjSDM}}.}
#' \item{family}{Response family.}
#' \item{time}{Runtime.}
#' \item{data}{List of Y, X (and spatial) model matrices.}
#' \item{sessionInfo}{Output of \code{\link{sessionInfo}}.}
#' \item{weights}{List of model coefficients (environmental (and spatial)).}
#' \item{sigma}{Lower triangular weight matrix for the covariance matrix.}
#' \item{history}{History of iteration losses.}
#' \item{se}{Matrix of standard errors, if \code{se = FALSE} the field 'se' is \code{NULL}.}
#' 
#' Implemented S3 methods include \code{\link{summary.sjSDM}}, \code{\link{plot.sjSDM}}, \code{\link{print.sjSDM}}, \code{\link{predict.sjSDM}}, and \code{\link{coef.sjSDM}}. For other methods, see section 'See Also'.
#' 
#' 
#' @references 
#' Chen, D., Xue, Y., & Gomes, C. P. (2018). End-to-end learning for the deep multivariate probit model. arXiv preprint arXiv:1803.08591.
#' 
#' Pichler, M., & Hartig, F. (2021). A new joint species distribution model for faster and more accurate inference of species associations from big community data. Methods in Ecology and Evolution, 12(11), 2159-2173. 
#' 
#' @example /inst/examples/sjSDM-example.R
#' @seealso  \code{\link{getCor}},  \code{\link{getCov}}, \code{\link{getRE}}, \code{\link{update.sjSDM}}, \code{\link{sjSDM_cv}}, \code{\link{DNN}}, \code{\link{NN}}, \code{\link{re}}, \code{\link{plot.sjSDM}}, \code{\link{print.sjSDM}}, \code{\link{predict.sjSDM}}, \code{\link{coef.sjSDM}}, \code{\link{summary.sjSDM}}, \code{\link{simulate.sjSDM}}, \code{\link{getSe}}, \code{\link{anova.sjSDM}}, \code{\link{importance}}
#' 
#' @import checkmate mathjaxr
#' @author Maximilian Pichler
#' @export
sjSDM = function(Y = NULL, 
                 env = NULL,
                 biotic = bioticStruct(),
                 spatial = NULL,
                 family = stats::binomial("probit"),
                 iter = 100L, 
                 step_size = NULL,
                 learning_rate = 0.01, 
                 se = FALSE, 
                 sampling = 100L,
                 parallel = 0L, 
                 control = sjSDMControl(),
                 device = "cpu", 
                 dtype = "float32",
                 seed = 758341678,
                 verbose = TRUE) {
  
  assertMatrix(Y)
  assert(checkMatrix(env), checkDataFrame(env), checkClass(env, "DNN"), checkClass(env, "linear"))
  assert(checkClass(spatial, "DNN"), checkClass(spatial, "linear"), checkNull(spatial))
  assert_class(biotic, "bioticStruct")
  qassert(iter, c("X1[1,)"))
  qassert(step_size, c("X1[1,)", "0"))
  qassert(learning_rate, c("R1(0,)"))
  qassert(sampling, c("X1(1,)"))
  qassert(parallel, c("X1[0,)"))
  qassert(control, c("L"))
  qassert(device, c("S1", "X1[0,)", "I1[0,)"))
  qassert(dtype, "S1")
  
  seed = as.integer(seed)
  
  if( any(apply(Y, 2, function(x) var(x, na.rm = TRUE)) < .Machine$double.eps) ) warning("No variation in at least one of Y columns detected!")
  
  if(inherits(family, "character")) {
    if(family == "nbinom") {
      family = stats::poisson()
      family$family = "nbinom"
    }
  }
  
  family = check_family(family)
  
  check_module()
  
  out = list()
  
  if(is.numeric(device)) device = as.integer(device)
  
  if(device == "gpu") device = 0L
  
  device = check_device(device)
  
  if(is.matrix(env) || is.data.frame(env)) env = linear(data = env)
  
  if(length(spatial$nn))
    stop("NN() is not supported in the spatial formula yet", call. = FALSE)
  if(length(spatial$re))
    stop("random-effect bars are not supported in the spatial formula yet", call. = FALSE)
  if(any(vapply(c(env$nn, env$re), function(b) nrow(b$X), 1L) != nrow(env$X)))
    stop("the NN()/random blocks and the parametric design have different numbers of rows; ",
         "sjSDM_cv() and hand-built designs do not carry them yet", call. = FALSE)
  
  out$cl = match.call()
  out$formula = env$formula
  out$names = colnames(env$X)
  out$species = colnames(Y)
  
  ### settings ##
  if(is.null(biotic$df)) {
    
    biotic$df = max(5L, as.integer(floor(ncol(Y) / 2)))
    
  }
  if(is.null(step_size)) step_size = as.integer(floor(nrow(env$X) * 0.1))
  else step_size = as.integer(step_size)
  
  output = as.integer(ncol(Y))
  
  # the environmental blocks first, then the spatial ones; the parametric block is the config
  # object itself and the NN() blocks hang off it. The random blocks are not in `blocks`:
  # they carry no net, and se() and the env/spatial bindings index `blocks` positionally.
  n_env_net = 1L + length(env$nn)
  blocks = c(net_blocks(env, output, TRUE), net_blocks(spatial, output, FALSE))
  block_at = c(seq_len(n_env_net),
               if(!is.null(spatial)) n_env_net + length(env$re) + seq_len(1L + length(spatial$nn)))
  random = net_random(env, output, n_env_net)
  
  control$optimizer$params$lr = learning_rate
  link = family$link
  
  out$model_properties = list(blocks = blocks, n_env = n_env_net, random = random,
                              block_at = block_at, optimizer = control$optimizer,
                              scheduler = control$scheduler_boolean,
                              patience = control$scheduler_patience,
                              factor = control$lr_reduce_factor, mixed = control$mixed,
                              device = device, dtype = dtype, seed = seed)
  out$loss_properties = list(link = link, species = output, df = biotic$df,
                             alpha = if(link == "probit") 1.70169 else 1.0,
                             diag = biotic$diag, l1 = biotic$l1_cov, l2 = biotic$l2_cov,
                             reg_on_Cov = biotic$reg_on_Cov, reg_on_Diag = biotic$on_diag,
                             inverse = biotic$inverse)
  
  mp = out$model_properties
  lp = out$loss_properties
  out$get_model = function() sjsdm_model(mp, lp)
  model = out$get_model()
  
  if(se && length(c(env$nn, env$re)))
    warning("standard errors are conditional on the fitted NN()/random blocks and ignore their uncertainty", call. = FALSE)
  envX = config_X(env)
  spX = config_X(spatial)
  
  if(is.null(spatial)) {
    time = system.time({model$fit(envX, Y, batch_size = step_size, 
                                  epochs = as.integer(iter), parallel = parallel, 
                                  sampling = as.integer(sampling),
                                  early_stopping_training=control$early_stopping_training,
                                  verbose = verbose)})[3]
    out$logLik = force_r( model$logLik(envX, Y,batch_size = step_size,parallel = parallel) )
    if(se && !inherits(env, "DNN")) try({ out$se = t(abind::abind(force_r(model$se(envX, Y, batch_size = step_size, parallel = parallel)),along=0L)) })
  
  } else {
    time = system.time({model$fit(envX, Y=Y,SP=spX, batch_size = step_size, 
                                  epochs = as.integer(iter), parallel = parallel, 
                                  sampling = as.integer(sampling),
                                  early_stopping_training=control$early_stopping_training,
                                  verbose = verbose)})[3]
    out$logLik = force_r( model$logLik(envX, Y, SP=spX, batch_size = step_size,parallel = parallel) )
    if(se && !inherits(env, "DNN")) try({ out$se = t(abind::abind(force_r(model$se(envX, Y, SP=spX,batch_size = step_size, parallel = parallel, verbose = verbose)),along=0L)) })
    
  }

  if(inherits(env, "linear")) class(out) = c("sjSDM", "linear")
  if(inherits(env, "DNN")) {
    out$env_architecture = parse_nn(model$env)
    class(out) = c("sjSDM", "DNN")
  }
  
  if(length(env$nn))
    out$env_architecture = paste0(sapply(seq_along(env$nn), function(k)
      paste0("NN block ", k, ":\n", parse_nn(model$net$blocks[[k + 1L]]))), collapse = "")
  
  if(inherits(spatial, "DNN")) out$spatial_architecture = parse_nn(model$spatial)
  
  if(!is.null(spatial)) class(out) = c(class(out), "spatial")
    
  out$model = model
  out$settings = list(biotic = biotic, env = env, spatial = spatial,iter = iter, 
                      step_size = step_size,learning_rate = learning_rate, control = control, 
                      parallel = parallel,device = device, dtype = dtype, sampling = sampling,
                      se = se)
  out$family = family
  out$time = time
  out$data = list(X = env$X, Y = Y)
  out$sessionInfo = utils::sessionInfo()
  out$weights = force_r(model$env_weights)
  out$sigma = force_r(model$get_sigma)
  if(out$family$family$family== "nbinom" || out$family$family$family== "gaussian" ) out$theta = force_r(model$get_theta)
  out$history = force_r(model$history)
  out$spatial_weights = force_r(model$spatial_weights)
  out$spatial = spatial
  out$Null = NULL # ?????
  out$seed = seed
  out$version = sjsdm_backend_version
  out$state = sjsdm_state(model)
  if(torch::cuda_is_available()) torch::cuda_empty_cache()
  return(out)
}


#' Print a fitted sjSDM model
#' 
#' @param x a model fitted by \code{\link{sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' 
#' @return No return value
#' 
#' @export
print.sjSDM = function(x, ...) {
  cat("sjSDM model, see summary(model) for details \n")
}


#' Predict from a fitted sjSDM model
#' 
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param newdata newdata for predictions
#' @param SP spatial predictors (e.g. X and Y coordinates)
#' @param Y Known occurrences of species, must be a matrix of the original size, species to be predicted must consist of NAs
#' @param type raw or link
#' @param dropout use dropout for predictions or not, only supported for DNNs
#' @param marginal integrate the latent factor out of the prediction. The species associations
#'   are generated by an unobserved latent factor z, so a marginal prediction is
#'   \eqn{E_z[\mathrm{link}(\mu + z\Sigma')]} and not \eqn{\mathrm{link}(\mu)}, which is the
#'   prediction at \eqn{z = 0}. The two differ whenever the associations are non-zero: for the
#'   probit link the marginal is \eqn{\Phi(\mu / \sqrt{\mathrm{diag}(\Sigma\Sigma' + I)})}.
#'   The integral is approximated by Monte Carlo, so predictions are stochastic; raise
#'   \code{sampling} for more stable values. \code{marginal = FALSE} restores the behaviour of
#'   sjSDM <= 1.1.0 and is only there for reproducing older results.
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#'
#' @details
#' For a model with random-effect bars the prediction for a group the fit saw is taken at the posterior
#' mean of that group. A group it did not see - a level of the grouping factor that was not in the training
#' data, or \code{NA} - contributes no mean, and its group effect is integrated over the prior
#' instead. That widens the prediction rather than shifting it, and for the \code{logit} and
#' \code{linear} links the prior variance is added by sampling, so those predictions carry one more source
#' of Monte-Carlo noise.
#'
#' @return Matrix of predictions (sites by species)
#'
#' @example /inst/examples/predict-example.R
#'
#' @import checkmate
#' @export
predict.sjSDM = function(object, newdata = NULL, SP = NULL, Y = NULL, type = c("link", "raw"),
                         dropout = FALSE, marginal = TRUE, ...) {
  object = checkModel(object)
  
  assert( checkNull(newdata), checkMatrix(newdata), checkDataFrame(newdata) )
  assert( checkNull(SP), checkMatrix(newdata), checkDataFrame(newdata) )
  qassert( dropout, "B1")
  
  if(!is.null(Y)) {
    if(object$family$link == "count") warning("Conditional predictions are available for binomial response only")
    type = "raw"
  }
  
  if(inherits(object, "spatial")) assert_class(object, "spatial")
  
  type = match.arg(type)
  
  if(type == "raw") link = FALSE
  else link = TRUE
  

  
  if(inherits(object, "spatial")) {
    
    
    if(is.null(newdata)) {
      return(force_r( object$model$predict(newdata = config_X(object$settings$env), SP = config_X(object$spatial), link=link, dropout = dropout, marginal = marginal, ...)))
    } else {
      
      newdata = c(sjsdm_newdata(object$settings$env, newdata),
                  re_newdata(object$settings$env, newdata))
      sp = sjsdm_newdata(object$spatial, SP)
      
    }
    pred = force_r(object$model$predict(newdata = newdata, SP = sp, link=link, dropout = dropout, marginal = marginal, ...))
    
    
  } else {
    
    if(is.null(newdata)) {
      return(force_r(object$model$predict(newdata = config_X(object$settings$env), link=link, marginal = marginal, ...)))
    } else {
      newdata = c(sjsdm_newdata(object$settings$env, newdata),
                  re_newdata(object$settings$env, newdata))
    }
    pred = force_r(object$model$predict(newdata = newdata, link=link, dropout = dropout, marginal = marginal, ...))
  }
  
  if(!is.null(Y)) {
    args = list(...)
    sampling = if(is.null(args[["sampling"]])) object$settings$sampling else args[["sampling"]]
    pred = sjsdm_conditional(object$model, pred, Y, sampling)
  }


  return(pred)
}



#' Return coefficients from a fitted sjSDM model
#' 
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' 
#' @return Matrix of environmental coefficients or list of environmental and spatial coefficients for spatial models. 
#' 
#' @export
coef.sjSDM = function(object, ...) {
  if(inherits(object, "spatial")) {
    return(list(env=object$weights, spatial = object$spatial_weights))
  } else {
    return(object$weights)
  }
}


#' Post hoc calculation of standard errors
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param step_size batch size for stochastic gradient descent
#' @param parallel number of cpu cores for the data loader, only necessary for large datasets
#' 
#' @return The object passed to this function but the \code{object$se} field contains the standard errors now
#' 
#' @export
getSe = function(object, step_size = NULL, parallel = 0L){
  if(!inherits(object, "sjSDM")) stop("object must be of class sjSDM")
  if(inherits(object$settings$env, "DNN"))
    stop("standard errors are only available for linear environmental models", call. = FALSE)
  object = checkModel(object)
  if(is.null(step_size)) step_size = object$settings$step_size
  else step_size = as.integer(step_size)
  # the spatial predictors have to go in, otherwise the Hessian is taken for a model
  # without the spatial term
  if(length(c(object$settings$env$nn, object$settings$env$re)))
    warning("standard errors are conditional on the fitted NN()/random blocks and ignore their uncertainty", call. = FALSE)
  SP = if(inherits(object, "spatial")) config_X(object$spatial) else NULL
  object$se = t(abind::abind(object$model$se(config_X(object$settings$env), object$data$Y, SP = SP,
                                             batch_size = step_size, parallel = parallel),
                             along = 0L))
  return(object)
}

#' Return summary of a fitted sjSDM model
#' 
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#' 
#' @return The above matrix is silently returned.
#' 
#' @export
summary.sjSDM = function(object, ...) {

  out = list()
  
  cat("Family: ", object$family$family$family, "\n\n")
  cat("LogLik: ", -object$logLik[[1]], "\n")
  cat("Regularization loss: ", object$logLik[[2]], "\n\n")
  
  if(object$family$family$family == "nbinom") {
    disps = 1+(softplus(object$theta)+0.00001)
    out$disperion = disps
    cat("Dispersion parameters for nbinom", disps, "\n\n")
  }
  
  cov_m = getCov(object)
  cor_m = stats::cov2cor(cov_m)
  
  p_cor = round(cor_m, 3)
  # p_cor[upper.tri(p_cor)] = NULL # TODO - find out what can bet set that is numeric and doesn't show
  colnames(p_cor) = paste0("sp", 1:ncol(p_cor))
  rownames(p_cor) = colnames(p_cor)
  

  if(dim(p_cor)[1] < 50) {
    cat("Species-species correlation matrix: \n\n")
    kk = format(p_cor,nsmall = 4)
    kk[upper.tri(kk)] = ""
    kk = cbind(colnames(p_cor), kk)
    kk = (apply(kk, 1:2, function(i) paste0("\t", i)))
    kk[, ncol(kk)] = paste0(kk[, ncol(kk)], "\n")
    cat(paste0(t(kk), collapse = ""))
    cat("\n\n\n")
  }
  
  if(inherits(object, "spatial")) {
    if(inherits(object$spatial, "linear")) {
      cat("Spatial: \n")
      sp = t(object$spatial_weights[[1]])
      rownames(sp) = colnames(object$spatial$X)
      colnames(sp) = paste0("sp", 1:ncol(sp))
      print(sp)
    } 
    
    if(inherits(object$spatial, "DNN")) {
      cat("Spatial architecture:\n")
      cat(object$spatial_architecture)
    }
    
    cat("\n\n\n")
  }
  
  if(length(object$settings$env$re)) {
    cat("Random effects, Lambda Lambda' averaged over species: \n")
    for(r in getRE(object)) {
      cat("(", paste(r$terms, collapse = " + "), " | ", r$group, "), ", length(r$levels),
          " levels, ",
          if(r$loading == "shared") "shared loading" else paste0("df = ", r$df), "\n", sep = "")
      print(round(if(length(dim(r$cov)) == 3L) apply(r$cov, 2:3, mean) else r$cov, 4))
    }
    cat("\n\n")
  }
  
  if(inherits(object, "linear")) {
  
      coefs = coef.sjSDM(object)[[1]]
      if(inherits(coefs, "list")) coefs = coefs[[1]]
      env2 = t(coefs)
    
      env = data.frame(env2)
      if(is.null(object$species)) colnames(env) = paste0("sp", 1:ncol(env))
      else colnames(env) = object$species
      rownames(env) = object$names
    
      # TO DO: p-value parsing:
      if(!is.null(object$se)) {
        out$z = env2 / object$se
        out$P = 2*stats::pnorm(abs(out$z),lower.tail = FALSE)
        out$se = object$se
        
        coefmat = cbind(
          as.vector(as.matrix(env)),
          as.vector(as.matrix(out$se)),
          as.vector(as.matrix(out$z)),
          as.vector(as.matrix(out$P))
          )
        colnames(coefmat) = c("Estimate", "Std.Err", "Z value", "Pr(>|z|)")
        rownames(coefmat) = apply(expand.grid( rownames(env), colnames(env)), 1, function(n) paste0(n[2]," ", n[1]))
        stats::printCoefmat(coefmat, signif.stars = getOption("show.signif.stars"), digits = 3)
        out$coefmat = coefmat
      } else {
      
      cat("Coefficients (beta): \n\n")
      if(dim(env)[2] > 50) utils::head(env)
      else print(env)
      } 
      if(!is.null(object$env_architecture)) cat("\n", object$env_architecture, sep = "")
    }else {
    
    cat("Env architecture:\n")
    cat(object$env_architecture)
    env = coef.sjSDM(object)
    
  }

  out$coefs = env
  out$logLik = object$logLik
  out$sigma = object$sigma
  out$cov = cov_m
  out$names = list(species = if(is.null(object$species)) paste0("sp", 1:ncol(object$data$Y)) else object$species, env = object$names)
  return(invisible(out))
}


#' Generates simulations from sjSDM model
#'
#' Simulate nsim responses from the fitted model following a multivariate probit model. 
#' So currently only supported for \code{family = stats::binomial("probit")}
#'
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param nsim number of simulations
#' @param seed seed for random number generator
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#' 
#' @return Array of simulated species occurrences of dimension order (nsim, sites, species)
#' 
#' @importFrom stats simulate
#' @export
simulate.sjSDM = function(object, nsim = 1, seed = NULL, ...) {
  object = checkModel(object)
  if(!is.null(seed)) sjsdm_set_seed(seed)
  pred = predict(object)
  
  if(object$family$family$family == "binomial") {
    sim = apply(pred, 1:2, function(p) stats::rbinom(nsim, 1, p))
  } else if(object$family$family$family == "poisson") {
    sim = apply(pred, 1:2, function(p) stats::rpois(nsim, p))
  } else if(object$family$family$family == "nbinom") {
    theta = 1/(softplus(object$theta)+0.00001)
    
    sim = 
      lapply(1:nsim, function(n) {
          t(sapply(1:nrow(pred), function(i) {
              sapply(1:ncol(pred), function(j) {
                prob = theta[j] / (theta[j] +pred[i,j]) +0.00001
                stats::rnbinom(1, size = theta[j], prob = prob)
                })
            })
          )
        })
    sim = abind::abind(sim, along = 0L)
  }
  return(sim)
}



#' Extract negative-log-Likelihood from a fitted sjSDM model
#'
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param individual returns internal ll structure, mostly for internal useage
#' @param ... optional arguments passed to internal logLik function (only used if \code{individual=TRUE})
#' 
#' @details
#' For a model with random-effect bars this is the conditional log-likelihood at the posterior means of the
#' group effects, not the evidence lower bound and not a marginal likelihood. The Kullback-Leibler term
#' enters the training loss only and is visible through \code{object$history$train_l}. Values are therefore
#' not comparable with those of a model fitted without a bar, and neither are information criteria built
#' from them.
#' 
#' @return Numeric value or numeric matrix if individual is true.
#' 
#' @importFrom stats simulate
#' @export
logLik.sjSDM <- function(object, individual=FALSE,...){
  if(!individual) return(object$logLik[[1]])
  else {
    object = checkModel(object)
    if(!inherits(object, "spatial")) return(force_r(object$model$logLik(config_X(object$settings$env), object$data$Y, individual = TRUE, ...)))
    else return(force_r(object$model$logLik(config_X(object$settings$env), object$data$Y, config_X(object$spatial), individual = TRUE, ...)))
  }
}




#' Update and re-fit a model call
#' 
#' @param object of class 'sjSDM'
#' @param env_formula new environmental formula
#' @param spatial_formula new spatial formula
#' @param biotic new biotic config
#' @param env_blocks indices of the environmental blocks to keep in the predictor, 1 being the
#'   parametric block and the \code{\link{NN}} blocks following it in formula order. Every
#'   other block is removed from the sum and contributes exactly zero. Random-effect bars are
#'   not selected here, they are kept in every refit.
#' @param ... additional arguments
#' 
#' @return An S3 class of type 'sjSDM'. See \code{\link{sjSDM}} for more information.
#' @export
update.sjSDM = function(object, env_formula = NULL, spatial_formula = NULL, biotic = NULL,
                        env_blocks = NULL, ...) {
  
  mf = match.call()
  if(!is.null(env_formula)){
    m = match("env_formula", names(mf))
    if(inherits(mf[m]$env_formula,"name")) mf[m]$env_formula = eval(mf[m]$env_formula, envir = parent.env(environment()))
    env_formula = stats::as.formula(mf[m]$env_formula)
  } else {
    env_formula = object$settings$env$formula
  }
  
  if(!is.null(spatial_formula)){
    m = match("spatial_formula", names(mf))
    if(inherits(mf[m]$spatial_formula, "name")) mf[m]$spatial_formula = eval(mf[m]$spatial_formula, envir = parent.env(environment()))
    spatial_formula = stats::as.formula(mf[m]$spatial_formula)
  } else {
    spatial_formula = object$settings$spatial$formula
  }
  
  if(is.null(biotic)) {
    biotic = object$settings$biotic
  } 
  
  env = object$settings$env
  env$formula = env_formula
  env[c("X", "terms", "xlevels", "intercept", "nn", "re")] = design_blocks(env_formula, env$data)
  if(!is.null(env_blocks)) env = select_blocks(env, env_blocks)
  
  if(inherits(object, "spatial")) {
    spatial = object$settings$spatial
    spatial$formula = spatial_formula
    spatial[c("X", "terms", "xlevels", "intercept")] = design(spatial_formula, spatial$data)
  } else {
    spatial = NULL
  }
  
  new_model = sjSDM(object$data$Y, 
                    env = env, 
                    spatial = spatial,
                    biotic = biotic,
                    family = object$family$family, 
                    iter = object$settings$iter, 
                    step_size = object$settings$step_size, 
                    learning_rate = object$settings$learning_rate,
                    sampling = object$settings$sampling,
                    control = object$settings$control,
                    device = object$settings$device, 
                    dtype = object$settings$dtype,
                    se = object$settings$se,
                    seed = object$seed,
                    ...
  )
  return(new_model)
}


#' Residuals for a sjSDM model
#'
#' Returns residuals for a fitted sjSDM model
#'
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param type residual type. Currently only supports raw
#' @param ... further arguments, not supported yet.
#' @return residuals in the format of the provided community matrix
#' 
#' @export
residuals.sjSDM <- function(object, type = "raw", ...){
  raw = object$data$Y - predict(object, type = "raw")
  return(raw)
}


