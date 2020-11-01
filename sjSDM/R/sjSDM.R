#' @title sjSDM
#'
#' @description fast and accurate joint species model
#' 
#' @param Y matrix of species occurences/responses in range [0,1]
#' @param env matrix of environmental predictors, object of type \code{\link{linear}} or \code{\link{DNN}}
#' @param biotic defines biotic (species-species associations) structure, object of type \code{\link{bioticStruct}}
#' @param spatial defines spatial structure, object of type \code{\link{linear}} or \code{\link{DNN}}
#' @param family error distribution with link function, see details for supported family functions
#' @param iter number of fitting iterations
#' @param step_size batch size for stochastic gradient descent, if \code{NULL} then step_size is set to: \code{step_size = 0.1*nrow(X)}
#' @param learning_rate learning rate for Adamax optimizer
#' @param se calculate standard errors for environmental coefficients
#' @param sampling number of sampling steps for Monte Carlo integreation
#' @param parallel number of cpu cores for the data loader, only necessary for large datasets
#' @param control control parameters for optimizer, see \code{\link{sjSDMControl}}
#' @param device which device to be used, "cpu" or "gpu"
#' @param dtype which data type, most GPUs support only 32 bit floats.
#' 
#' @details The function fits a multivariate probit model via Monte-Carlo integration (see Chen et al., 2018) of the joint likelihood for all species. See Pichler and Hartig, 2020 for benchmark results.
#' 
#' sjSDM depends on the anaconda python distribution and pytorch, which need to be installed before being able to use the sjSDM function. 
#' See \code{\link{install_sjSDM}}, \code{vignette("Dependencies", package = "sjSDM")}
#' 
#' @section Family:
#' Currently supported distributions and link functions:
#' \itemize{
#' \item \code{\link{binomial}}: \code{"probit"} or \code{"logit"}
#' \item \code{\link{poisson}}: \code{"log"} 
#' \item \code{\link{gaussian}}: \code{"identity"} 
#' }
#' 
#' @section Installation:
#' \code{\link{install_sjSDM}} should be theoretically able to install conda and pytorch automatically. If \code{\link{sjSDM}} still does not work after reloading RStudio, you can try to solve this in on your own with our trouble shooting guide \code{\link{installation_help}}.
#' If the trouble shooting guide did not help, please create an issue on \href{https://github.com/TheoreticalEcology/s-jSDM/issues}{issue tracker} with a copy of the \code{\link{install_diagnostic}} output as a quote. 
#' 
#' @references 
#' Chen, D., Xue, Y., & Gomes, C. P. (2018). End-to-end learning for the deep multivariate probit model. arXiv preprint arXiv:1803.08591.
#' 
#' Pichler, M., and Hartig, F. (2020). A new method for faster and more accurate inference of species associations from novel community data. arXiv preprint arXiv:2003.05331.
#' 
#' 
#' @example /inst/examples/sjSDM-example.R
#' @seealso \code{\link{sjSDM_cv}}, \code{\link{DNN}}, \code{\link{print.sjSDM}}, \code{\link{predict.sjSDM}}, \code{\link{coef.sjSDM}}, \code{\link{summary.sjSDM}}, \code{\link{getCov}}, \code{\link{simulate.sjSDM}}, \code{\link{getSe}}, \code{\link{anova.sjSDM}}, \code{\link{importance}}
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
                 sampling = 1000L,
                 parallel = 0L, 
                 control = sjSDMControl(),
                 device = "cpu", 
                 dtype = "float32") {
  stopifnot(
    !is.null(Y),
    iter >= 0,
    learning_rate >= 0#,
    #!max(Y) > 1.0,
    #!min(Y) < 0.0
  )
  
  family = check_family(family)
  
  check_module()
  
  out = list()
  
  if(is.numeric(device)) device = as.integer(device)
  
  if(device == "gpu") device = 0L
  
  if(is.matrix(env) || is.data.frame(env)) env = linear(data = env)
  
  out$formula = env$formula
  out$names = colnames(env$X)
  out$species = colnames(Y)
  out$cl = match.call()
  
  ### settings ##
  if(is.null(biotic$df)) {
    
    biotic$df = max(5L, as.integer(floor(ncol(Y) / 2)))
    
  }
  if(is.null(step_size)) step_size = as.integer(floor(nrow(env$X) * 0.1))
  else step_size = as.integer(step_size)
  
  output = as.integer(ncol(Y))
  input = as.integer(ncol(env$X))
  
  
  out$get_model = function(){
    model = fa$Model_sjSDM( device = device, dtype = dtype)
    
    if(inherits(env, "DNN")) {
      activation=env$activation
      hidden = as.integer(env$hidden)
    } else {
      hidden = list()
      activation = c("linear")
    }
    
    if(!is.null(env$dropout)) dropout_env = env$dropout
    else dropout_env = -99
    
    model$add_env(input, output, hidden = hidden, activation = activation, l1 = env$l1, l2=env$l2, dropout=dropout_env)
    
    if(!is.null(spatial)) {
      
      if(!is.null(spatial$dropout)) dropout_sp = spatial$dropout
      else dropout_sp = -99
      
      if(inherits(spatial, "DNN")) {
        activation_spatial=spatial$activation
        hidden_spatial = spatial$hidden
        model$add_spatial(as.integer(ncol(spatial$X)), output_shape = output, hidden = hidden_spatial, activation = activation_spatial, l1 = spatial$l1, l2= spatial$l2, dropout=dropout_sp)
      } 
      if(inherits(spatial, "linear")) {
        model$add_spatial(as.integer(ncol(spatial$X)), output_shape = output, l1 = spatial$l1, l2= spatial$l2)
      }
    }
    
    control$optimizer$params$lr = learning_rate
    optimizer = do.call(control$optimizer$ff(), control$optimizer$params)
    
    model$build(df = biotic$df, 
                l1 = biotic$l1_cov, 
                l2 = biotic$l2_cov, 
                reg_on_Diag = biotic$on_diag,
                inverse = biotic$inverse,
                reg_on_Cov = biotic$reg_on_Cov,
                optimizer = optimizer, 
                link = family$link,
                diag=biotic$diag,
                scheduler=control$schedule)
    
    return(model)
  }
  model = out$get_model()
  
  if(is.null(spatial)) {
    time = system.time({model$fit(env$X, Y, batch_size = step_size, epochs = as.integer(iter), parallel = parallel, sampling = as.integer(sampling))})[3]
    out$logLik = model$logLik(env$X, Y,batch_size = step_size,parallel = parallel)
    if(se && !inherits(env, "DNN")) try({ out$se = t(abind::abind(model$se(env$X, Y, batch_size = step_size, parallel = parallel),along=0L)) })
  
  } else {
    time = system.time({model$fit(env$X, Y=Y,SP=spatial$X, batch_size = step_size, epochs = as.integer(iter), parallel = parallel, sampling = as.integer(sampling))})[3]
    out$logLik = model$logLik(env$X, Y, SP=spatial$X, batch_size = step_size,parallel = parallel)
    if(se && !inherits(env, "DNN")) try({ out$se = t(abind::abind(model$se(env$X, Y, SP=spatial$X,batch_size = step_size, parallel = parallel),along=0L)) })
    
  }

  if(inherits(env, "linear")) class(out) = c("sjSDM", "linear")
  if(inherits(env, "DNN")) {
    out$env_architecture = parse_nn(model$env)
    class(out) = c("sjSDM", "DNN")
  }
  
  if(inherits(spatial, "DNN")) out$spatial_architecture = parse_nn(model$spatial)
  
  if(!is.null(spatial)) class(out) = c(class(out), "spatial")
    
  out$model = model
  out$settings = list(biotic = biotic, env = env, spatial = spatial,iter = iter, 
                      step_size = step_size,learning_rate = learning_rate, 
                      parallel = parallel,device = device, dtype = dtype, sampling = sampling)
  out$family = family
  out$time = time
  out$data = list(X = env$X, Y = Y)
  out$sessionInfo = utils::sessionInfo()
  out$weights = model$env_weights
  out$sigma = model$get_sigma
  out$history = model$history
  out$spatial_weights = model$spatial_weights
  out$spatial = spatial
  torch$cuda$empty_cache()
  return(out)
}


#' Print a fitted sjSDM model
#' 
#' @param x a model fitted by \code{\link{sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
print.sjSDM = function(x, ...) {
  cat("sjSDM model, see summary(model) for details \n")
}


#' Predict from a fitted sjSDM model
#' 
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param newdata newdata for predictions
#' @param SP spatial predictors (e.g. X and Y coordinates)
#' @param type raw or link
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
predict.sjSDM = function(object, newdata = NULL, SP = NULL, type = c("link", "raw"),...) {
  object = checkModel(object)
  
  type = match.arg(type)
  
  if(type == "raw") link = FALSE
  else link = TRUE
  
  if(inherits(object, "spatial")) {
    
    
    if(is.null(newdata)) {
      return(object$model$predict(newdata = object$data$X, SP = object$spatial$X, link=link))
    } else {
      
      if(is.data.frame(newdata)) {
        newdata = stats::model.matrix(object$formula, newdata)
      } else {
        newdata = stats::model.matrix(object$formula, data.frame(newdata))
      }
      
      if(is.data.frame(SP)) {
        sp = stats::model.matrix(object$spatial$formula, SP)
      } else {
        sp = stats::model.matrix(object$spatial$formula, data.frame(SP))
      }
      
    }
    pred = object$model$predict(newdata = newdata, SP = sp, link=link, ...)
    return(pred)
    
    
  } else {
    
    if(is.null(newdata)) {
      return(object$model$predict(newdata = object$data$X, link=link))
    } else {
      if(is.data.frame(newdata)) {
        newdata = stats::model.matrix(object$formula, newdata)
      } else {
        newdata = stats::model.matrix(object$formula, data.frame(newdata))
      }
    }
    pred = object$model$predict(newdata = newdata, link=link, ...)
    return(pred)
    
  }
}



#' Return coefficients from a fitted sjSDM model
#' 
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
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
#' @export
getSe = function(object, step_size = NULL, parallel = 0L){
  if(!inherits(object, "sjSDM")) stop("object must be of class sjSDM")
  object = checkModel(object)
  if(is.null(step_size)) step_size = object$settings$step_size
  else step_size = as.integer(step_size)
  if(!inherits(object, "spatialRE")) try({ object$se = t(abind::abind(object$model$se(object$data$X, object$data$Y, batch_size = step_size, parallel = parallel),along=0L)) })
  else try({ object$se = t(abind::abind(object$model$se(object$data$X, object$data$Y, object$spatial$re, batch_size = step_size, parallel = parallel),along=0L)) })
  return(object)
}

#' Return summary of a fitted sjSDM model
#' 
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#' @export
summary.sjSDM = function(object, ...) {

  out = list()
  
  cat("LogLik: ", -object$logLik[[1]], "\n")
  cat("Deviance: ", 2*object$logLik[[1]], "\n\n")
  cat("Regularization loss: ", object$logLik[[2]], "\n\n")
  
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
  
  
  if(inherits(object, "linear")) {
  
      coefs = coef.sjSDM(object)[[1]]
      if(inherits(coefs, "list")) coefs = coefs[[1]]
      env2 = t(coefs)
    
      env = data.frame(env2)
      if(is.null(object$species)) colnames(env) = paste0("sp", 1:ncol(env))
      else colnames(env) = object$species
      rownames(env) = object$names
    
      
      # if(inherits(object, "spatialRE")) {
      #   cat("Spatial random effects (Intercept): \n")
      #   cat("\tVar: ", round(stats::var(object$re), 3), "\n\tStd. Dev.: ", round(stats::sd(object$re), 3), "")
      #   cat("\n\n\n")
      # }
      # 
      
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
    }else {
    
    cat("Env architecture:\n")
    cat(object$env_architecture)
    env = coef.sjSDM(object)
    
  }

  out$coefs = env
  out$logLik = object$logLik
  out$sigma = object$sigma
  out$cov = cov_m
  return(invisible(out))
}


#' Generates simulations from sjSDM model
#'
#' Simulate nsim responses from the fitted model
#'
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param nsim number of simulations
#' @param seed seed for random numer generator
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#'
#' @importFrom stats simulate
#' @export
simulate.sjSDM = function(object, nsim = 1, seed = NULL, ...) {
  object = checkModel(object)
  if(!is.null(seed)) {
    set.seed(seed)
    torch$cuda$manual_seed(seed)
    torch$manual_seed(seed)
  }
  preds = abind::abind(lapply(1:nsim, function(i) predict.sjSDM(object)), along = 0L)
  simulation = apply(preds, 2:3, function(p) stats::rbinom(nsim, 1L,p))
  return(simulation)
}



#' Extract Log-Likelihood from a fitted sjSDM model
#'
#' @param object a model fitted by \code{\link{sjSDM}}
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#'
#' @importFrom stats simulate
#' @export
logLik.sjSDM <- function(object, ...){
  return(object$logLik[[1]])
}
