#' Cross validation of elastic net tuning
#' 
#' @param Y species occurrence matrix
#' @param env matrix of environmental predictors or object of type \code{\link{linear}}, or \code{\link{DNN}}
#' @param biotic defines biotic (species-species associations) structure, object of type \code{\link{bioticStruct}}. Alpha and lambda have no influence
#' @param spatial defines spatial structure, object of type \code{\link{linear}}, or \code{\link{DNN}}
#' @param tune tuning strategy, random or grid search
#' @param tune_steps number of tuning steps
#' @param CV n-fold cross validation
#' @param alpha_cov weighting of l1 and l2 on covariances: \eqn{(1 - \alpha) * |cov| + \alpha ||cov||^2}
#' @param alpha_coef weighting of l1 and l2 on coefficients: \eqn{(1 - \alpha) * |coef| + \alpha ||coef||^2}
#' @param alpha_spatial weighting of l1 and l2 on spatial coefficients: \eqn{(1 - \alpha) * |coef_sp| + \alpha ||coef_sp||^2}
#' @param lambda_cov overall regularization strength on covariances
#' @param lambda_coef overall regularization strength on coefficients
#' @param lambda_spatial overall regularization strength on spatial coefficients
#' @param device device, default cpu
#' @param n_cores number of cores for parallelization 
#' @param n_gpu number of GPUs
#' @param sampling number of sampling steps for Monte Carlo integration
#' @param blocks blocks of parallel tuning steps
#' @param ... arguments passed to sjSDM, see \code{\link{sjSDM}}
#' 
#' @return 
#' An S3 class of type 'sjSDM_cv' including the following components:
#' 
#' \item{tune_results}{Data frame with tuning results.}
#' \item{short_summary}{Data frame with averaged tuning results.}
#' \item{summary}{Data frame with summarized averaged results.}
#' \item{settings}{List of tuning settings, see the arguments in \code{\link{DNN}}.}
#' \item{data}{List of Y, env (and spatial) objects.}
#' \item{config}{List of \code{\link{sjSDM}} settings, see arguments of \code{\link{sjSDM}}.}
#' \item{spatial}{Logical, spatial model or not.}
#' 
#' Implemented S3 methods include \code{\link{sjSDM.tune}}, \code{\link{plot.sjSDM_cv}}, \code{\link{print.sjSDM_cv}}, and \code{\link{summary.sjSDM_cv}}
#' 
#' @example /inst/examples/sjSDM_cv-example.R
#' @seealso \code{\link{plot.sjSDM_cv}}, \code{\link{print.sjSDM_cv}}, \code{\link{summary.sjSDM_cv}}, \code{\link{sjSDM.tune}}
#' @import checkmate
#' @export

sjSDM_cv = function(Y, 
                    env = NULL, 
                    biotic = bioticStruct(), 
                    spatial = NULL, 
                    tune = c("random", "grid"), 
                    CV = 5L, 
                    tune_steps = 20L,
                    alpha_cov = seq(0.0, 1.0, 0.1), 
                    alpha_coef = seq(0.0, 1.0, 0.1),
                    alpha_spatial = seq(0.0, 1.0, 0.1), 
                    lambda_cov = 2^seq(-10,-1, length.out = 20),
                    lambda_coef = 2^seq(-10,-0.5, length.out = 20),
                    lambda_spatial = 2^seq(-10,-0.5, length.out = 20),
                    device="cpu",
                    n_cores = NULL, 
                    n_gpu = NULL,
                    sampling = 5000L,
                    blocks = 1L,
                    ...) {
  
  assertMatrix(Y)
  assert(checkMatrix(env), checkDataFrame(env), checkClass(env, "DNN"), checkClass(env, "linear"))
  assert(checkClass(spatial, "DNN"), checkClass(spatial, "linear"), checkNull(spatial))
  assert_class(biotic, "bioticStruct")
  qassert(CV, "X1[1,)")
  qassert(tune_steps, "X1[1,)")
  qassert(alpha_cov, "R+[0,)")
  qassert(alpha_coef, "R+[0,)")
  qassert(alpha_spatial, "R+[0,)")
  qassert(lambda_cov, "R+[0,)")
  qassert(lambda_coef, "R+[0,)")
  qassert(lambda_spatial, "R+[0,)")
  qassert(device, c("S1", "X1[0,)", "I1[0,)"))
  qassert(n_cores, c("X1[1,)", "0"))
  qassert(n_gpu, c("X+[0,)", "0"))
  qassert(sampling, "X1[0,)")
  qassert(blocks, "X1[1,)")
  
  ellip = list(...)
  
  tune = match.arg(tune)
  if(is.matrix(env) || is.data.frame(env)) env = linear(data = env)
  if(is.matrix(spatial) || is.data.frame(spatial)) spatial = linear(data = spatial)
  
  set = cut(sample.int(nrow(env$X)), breaks = CV, labels = FALSE)
  test_indices = lapply(unique(set), function(s) which(set == s, arr.ind = TRUE))
  
  if(is.null(spatial)) {
    if(tune == "random") { 
      tune_samples = data.frame(t(sapply(1:tune_steps, function(i) c(sample(alpha_cov, 1), 
                                                                  sample(alpha_coef, 1),
                                                                  sample(lambda_cov, 1),
                                                                  sample(lambda_coef, 1)))))
    } else {
      tune_samples = expand.grid(alpha_cov, alpha_coef, lambda_cov, lambda_coef)
    }
    colnames(tune_samples) = c("alpha_cov", "alpha_coef", "lambda_cov", "lambda_coef")
  } else {
    if(tune == "random") { 
      tune_samples = data.frame(t(sapply(1:tune_steps, function(i) c(sample(alpha_cov, 1), 
                                                                  sample(alpha_coef, 1),
                                                                  sample(alpha_spatial, 1),
                                                                  sample(lambda_cov, 1),
                                                                  sample(lambda_coef, 1),
                                                                  sample(lambda_spatial, 1)))))
    } else {
      tune_samples = expand.grid(alpha_cov, alpha_coef,alpha_spatial, lambda_cov, lambda_coef, lambda_spatial)
    }
    colnames(tune_samples) = c("alpha_cov", "alpha_coef", "alpha_spatial","lambda_cov", "lambda_coef", "lambda_spatial")
  }

  
  tune_func = function(t){

    if(!is.null(spatial)) {
      a_cov = tune_samples[t,1]
      a_coef = tune_samples[t,2]
      a_sp = tune_samples[t,3]
      l_cov = tune_samples[t,4]
      l_coef = tune_samples[t,5]
      l_sp = tune_samples[t,6]
    } else {
      a_cov = tune_samples[t,1]
      a_coef = tune_samples[t,2]
      l_cov = tune_samples[t,3]
      l_coef = tune_samples[t,4]
    }
    

    # lists work better for parallel support 
    cv_step_result = vector("list", CV)
    for(i in 1:length(test_indices)) {
      ### model ###
      new_env = env
      X_test = env$X[test_indices[[i]],,drop = FALSE]
      Y_test = Y[test_indices[[i]],,drop=FALSE]
      
      new_env$X = env$X[-test_indices[[i]],,drop = FALSE]
      Y_train = Y[-test_indices[[i]],,drop=FALSE]
      
      new_env$l1_coef = (1-a_coef)*l_coef
      new_env$l2_coef = (a_coef)*l_coef
      biotic$l1_cov =  (1-a_cov)*l_cov
      biotic$l2_cov =  (a_cov)*l_cov
      new_env$formula = stats::as.formula("~0+.")

      if(!is.null(spatial)) {
        new_spatial = spatial
        SP_test = spatial$X[test_indices[[i]],,drop = FALSE]
        new_spatial$X = spatial$X[-test_indices[[i]],,drop = FALSE]
        new_spatial$l1_coef = (1-a_sp)*l_sp
        new_spatial$l2_coef = (a_sp)*l_sp
        new_spatial$formula = stats::as.formula("~0+.")
      } else {
        new_spatial = NULL
        SP_test = NULL
      }
      
      if(!is.null(n_gpu)) {
          
        myself = paste(Sys.info()[['nodename']], Sys.getpid(), sep='-')
        
                 
         if(length(n_gpu) == 1) dist = cbind(nodes,(n_gpu-1):0)
         else dist = cbind(nodes,n_gpu)
         device2 = as.integer(as.numeric(dist[which(dist[,1] %in% myself, arr.ind = TRUE), 2]))
         model = sjSDM(Y = Y_train, env = new_env, biotic = biotic, spatial = new_spatial,device=device2,sampling=sampling, ...)
      } else {
        model = sjSDM(Y = Y_train, env = new_env, biotic = biotic, spatial = new_spatial,device=device, sampling=sampling,...)
      }
      
      mean_func = function(f) apply(abind::abind(lapply(1:50, function(i) f() ),along = -1), 2:3, mean)
      
      pred_test = mean_func(function() predict.sjSDM(model, newdata = X_test, SP = SP_test) )
      pred_train = mean_func(function() predict.sjSDM(model) )
      auc_test = sapply(1:ncol(Y_test), function(s) {
        a = Metrics::auc(Y_test[,s], pred_test[,s])
        return(ifelse(is.na(a),0.5, a))} )
      auc_train = sapply(1:ncol(Y_train), function(s) {
        a = Metrics::auc(Y_train[,s], pred_train[,s])
        return(ifelse(is.na(a),0.5, a))
        })
      auc_macro_test = sum(auc_test * (apply(Y_test, 2, sum)/sum(Y_test)))
      auc_macro_train = sum(auc_train * (apply(Y_train, 2, sum)/sum(Y_train)))
      auc_test = mean(auc_test)
      auc_train = mean(auc_train)
      ll_train = logLik.sjSDM(model)
      bs =  as.integer(model$settings$step_size)
      if(bs > nrow(X_test)) bs = 1L
      ll_test =  mean(sapply(1:20, function(i) force_r( model$model$logLik(X_test, Y_test, SP = SP_test, batch_size =bs, sampling=sampling) )[[1]] ))
      cv_step_result[[i]] = list(indices = test_indices[[i]], 
                                 pars = tune_samples[t,],
                                 pred_test = pred_test,
                                 pred_train = pred_train,
                                 ll_train = ll_train,
                                 ll_test = ll_test,
                                 auc_test = auc_test,
                                 auc_train = auc_train,
                                 auc_macro_test = auc_macro_test,
                                 auc_macro_train = auc_macro_train)
      rm(model)
      pkg.env$torch$cuda$empty_cache()
    }
    return(cv_step_result)
  }
  
  
  if(is.null(n_cores)) {
    result = vector("list", nrow(tune_samples))
    for(t in 1:nrow(tune_samples)){
      result[[t]] = tune_func(t)
    }
  } else {

    blocks_run = cut(1:nrow(tune_samples), ceiling(nrow(tune_samples)/blocks))
    result_list = vector("list", length(unique(blocks_run)))
    
    cl = parallel::makeCluster(n_cores)
    nodes = unlist(parallel::clusterEvalQ(cl, paste(Sys.info()[['nodename']], Sys.getpid(), sep='-')))
    #print(nodes)
    control = parallel::clusterEvalQ(cl, {library(sjSDM)})
    if(length(ellip) > 0 ) parallel::clusterExport(cl, list("tune_samples", "test_indices","biotic", "CV", "env","spatial", "Y", "nodes","n_gpu","n_cores","device","sampling","..."), envir = environment())
    else parallel::clusterExport(cl, list("tune_samples", "test_indices","biotic", "CV", "env","spatial", "Y", "nodes","n_gpu","n_cores","device","sampling"), envir = environment())
    
    for(i in 1:length(unique(blocks_run))){
      ind = blocks_run == unique(blocks_run)[i]
      sub_tune_samples = tune_samples[ind, ]
      
      result_list[[i]] = parallel::parLapply(cl, 1:nrow(sub_tune_samples), tune_func)
    }
    result = do.call(rbind, result_list)
    parallel::stopCluster(cl)
    
  }
  summary_results = 
    data.frame(do.call(rbind, lapply(1:CV, function(i) tune_samples)), 
               iter = rep(1:nrow(tune_samples), CV),
               CV_set = sort(rep(1:CV, nrow(tune_samples))), 
               ll_train = rep(NA, CV*nrow(tune_samples)),
               ll_test = rep(NA, CV*nrow(tune_samples)),
               AUC_test = rep(NA, CV*nrow(tune_samples)),
               AUC_macro_test = rep(NA, CV*nrow(tune_samples)),
               AUC_train = rep(NA, CV*nrow(tune_samples)),
               AUC_macro_train = rep(NA, CV*nrow(tune_samples)))
  
  if(is.null(spatial)) n=7
  else n=9
  for(t in 1:nrow(tune_samples)) {
    for(i in 1:CV) {
      summary_results[summary_results$iter == t & summary_results$CV_set == i, n] = result[[t]][[i]]$ll_train
      summary_results[summary_results$iter == t & summary_results$CV_set == i, n+1] = result[[t]][[i]]$ll_test
      summary_results[summary_results$iter == t & summary_results$CV_set == i, n+2] = result[[t]][[i]]$auc_test
      summary_results[summary_results$iter == t & summary_results$CV_set == i, n+3] = result[[t]][[i]]$auc_macro_test
      summary_results[summary_results$iter == t & summary_results$CV_set == i, n+4] = result[[t]][[i]]$auc_train
      summary_results[summary_results$iter == t & summary_results$CV_set == i, n+5] = result[[t]][[i]]$auc_macro_test
    }
  }
  
  res = data.frame(tune_step = 1:nrow(tune_samples), 
                   AUC_test = rep(NA,nrow(tune_samples)),
                   AUC_test_macro = rep(NA,nrow(tune_samples)),
                   logLik = rep(NA,nrow(tune_samples)))
  
  for(t in 1:nrow(tune_samples)) {
      res[t,2] = mean(summary_results$AUC_macro_test[summary_results$iter == t])
      res[t,3] = mean(summary_results$AUC_train[summary_results$iter == t])
      res[t,4] = sum(summary_results$ll_test[summary_results$iter == t])
  }
  short_summary = cbind(tune_samples, res[,-1])
  short_summary$l1_cov = (1-short_summary$alpha_cov)*short_summary$lambda_cov
  short_summary$l2_cov = short_summary$alpha_cov*short_summary$lambda_cov
  short_summary$l1_coef = (1-short_summary$alpha_coef)*short_summary$lambda_coef
  short_summary$l2_coef = short_summary$alpha_coef*short_summary$lambda_coef

  if(!is.null(spatial)) {
    short_summary$l1_sp = (1-short_summary$alpha_sp)*short_summary$lambda_sp
    short_summary$l2_sp = short_summary$alpha_sp*short_summary$lambda_sp
  }
  out = list(tune_results = result, 
             short_summary = short_summary, 
             summary = summary_results, 
             settings = list(tune_samples = tune_samples, CV = CV, tune = tune),
             data = list(Y = Y, env = env, biotic = biotic, spatial = spatial),
             config = c(list(device=device, sampling=sampling), as.list(unlist(ellip))),
             spatial = !is.null(spatial))
  class(out) = c("sjSDM_cv")
  return(out)
}

zero_one = function(x) (x-min(x))/(max(x) - min(x))

#' @rdname sjSDM
#' @param object object of type \code{\link{sjSDM_cv}}
#' @author Maximilian Pichler
#' 
#' @return 
#' 
#' \code{\link{sjSDM.tune}} returns an S3 object of class 'sjSDM', see above for information about values.
#' 
#' @export
sjSDM.tune = function(object) {
  if(!inherits(object, "sjSDM_cv")) stop("Object must be of type sjSDM_cv, see function ?sjSDM_cv")
  x = object$short_summary
  maxP = which.min(x[["logLik"]])
  
  if(object$spatial) {
    best_values = 
      c(lambda_cov = x[maxP,]$lambda_cov,
        alpha_cov = x[maxP,]$alpha_cov,
        lambda_coef =  x[maxP,]$lambda_coef,
        alpha_coef = x[maxP,]$alpha_coef,
        lambda_spatial =  x[maxP,]$lambda_spatial,
        alpha_spatial = x[maxP,]$alpha_spatial)
  } else {
    best_values = 
      c(lambda_cov = x[maxP,]$lambda_cov,
        alpha_cov = x[maxP,]$alpha_cov,
        lambda_coef =  x[maxP,]$lambda_coef,
        alpha_coef = x[maxP,]$alpha_coef)
  }
  spatial = object$data$spatial
  env = object$data$env
  biotic = object$data$biotic
  settings = object$config
  
  ## update env regularization ##
  lambda_coef = best_values[["lambda_coef"]]
  if(lambda_coef == 0.0) {
    lambda_coef = -99.9
  }
  env$l1_coef = (1-best_values[["alpha_coef"]])*lambda_coef
  env$l2_coef = best_values[["alpha_coef"]]*lambda_coef
  
  ## update biotic regularization ##
  biotic$l1_cov = (1-best_values[["alpha_cov"]])*best_values[["lambda_cov"]]
  biotic$l2_cov = best_values[["alpha_cov"]]*best_values[["lambda_cov"]]
  
  ## update space ##
  if(object$spatial) {
    lambda_spatial = best_values[["lambda_spatial"]]
    if(lambda_spatial == 0.0) {
      lambda_spatial = -99.9
    }
    spatial$l1_coef = (1-best_values[["alpha_spatial"]])*lambda_spatial
    spatial$l2_coef = best_values[["alpha_spatial"]]*lambda_spatial
  }
  
  settings = c(list(Y = object$data$Y,
                    env = env,
                    spatial = spatial,
                    biotic = biotic),
               settings
               )
  model = do.call(sjSDM, settings)
  return(model)
}


#' Print a fitted sjSDM_cv model
#' 
#' @param x a model fitted by \code{\link{sjSDM_cv}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' 
#' @return Above data frame is silently returned.
#' @export
print.sjSDM_cv = function(x, ...) {
  print(x$summary)
  return(invisible(x$summary))
}


#' Return summary of a fitted sjSDM_cv model
#' 
#' @param object a model fitted by \code{\link{sjSDM_cv}}
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#' @return Above data frame is silently returned.
#' @export
summary.sjSDM_cv = function(object, ...) {
  print(object$short_summary)
  return(invisible(object$short_summary))
}


#' Plot elastic net tuning
#' 
#' @param x a model fitted by \code{\link{sjSDM_cv}}
#' @param y unused argument
#' @param perf performance measurement to plot
#' @param resolution resolution of grid
#' @param k number of knots for the gm
#' @param ... Additional arguments to pass to \code{plot()}
#' 
#' @return 
#' Named vector of optimized regularization parameters.
#' 
#' Without space:
#' 
#' \item{lambda_cov}{Regularization strength in the \code{\link{bioticStruct}} object.}
#' \item{alpha_cov}{Weigthing between L1 and L2 in the \code{\link{bioticStruct}} object.}
#' \item{lambda_coef}{Regularization strength in the \code{\link{linear}} or \code{\link{DNN}} object.}
#' \item{alpha_coef}{Weigthing between L1 and L2 in the \code{\link{linear}} or \code{\link{DNN}} object.}
#' 
#' With space:
#' 
#' \item{lambda_cov}{Regularization strength in the \code{\link{bioticStruct}} object.}
#' \item{alpha_cov}{Weigthing between L1 and L2 in the \code{\link{bioticStruct}} object.}
#' \item{lambda_coef}{Regularization strength in the \code{\link{linear}} or \code{\link{DNN}} object.}
#' \item{alpha_coef}{Weigthing between L1 and L2 in the \code{\link{linear}} or \code{\link{DNN}} object.}
#' \item{lambda_spatial}{Regularization strength in the \code{\link{linear}} or \code{\link{DNN}} object for the spatial component.}
#' \item{alpha_spatial}{Weigthing between L1 and L2 in the\code{\link{linear}} or \code{\link{DNN}} object for the spatial component.}
#' 
#' @export
plot.sjSDM_cv = function(x, y, perf = c("logLik", "AUC", "AUC_macro"), resolution = 6,k = 3, ...) {
  oldpar = par(no.readonly = TRUE)
  on.exit(par(oldpar))
  x = x$short_summary
  perf = match.arg(perf)
  if(perf == "AUC") perf = "AUC_test"
  if(perf == "AUC_macro") perf = "AUC_test_macro"
  
  if("lambda_spatial"  %in% colnames(x)) spatial = TRUE
  else spatial = FALSE
  
  if(spatial) form = paste0(perf, " ~ te(alpha_cov, lambda_cov, k = ",k,") + te(alpha_coef, lambda_coef, k = ", k, ") + te(alpha_spatial, lambda_spatial, k = ", k ,")")
  else form = paste0(perf, " ~ te(alpha_cov, lambda_cov, k = ",k,") + te(alpha_coef, lambda_coef, k = ", k, ")")
  g = mgcv::gam(stats::as.formula(form), data = x)
  
  xn1 = seq(min(x$alpha_cov), max(x$alpha_cov), length.out = resolution)
  yn1 = seq(min(x$lambda_cov), max(x$lambda_cov), length.out = resolution)
  xn2 = seq(min(x$alpha_coef), max(x$alpha_coef), length.out = resolution)
  yn2 = seq(min(x$lambda_coef), max(x$lambda_coef), length.out = resolution)
  if(spatial) {
    
    xn3 = seq(min(x$alpha_spatial), max(x$alpha_spatial), length.out = resolution)
    yn3 = seq(min(x$lambda_spatial), max(x$lambda_spatial), length.out = resolution)
    
    d = data.frame(expand.grid(xn1, yn1, xn2, yn2, xn3, yn3))
    colnames(d) = c("alpha_cov", "lambda_cov", "alpha_coef", "lambda_coef", "alpha_spatial", "lambda_spatial")
    pp = mgcv::predict.gam(g, d)
    preds = cbind(pp, d)
    
    res_coef = vector("list", resolution^4)
    res_cov = vector("list", resolution^4)
    res_spatial = vector("list", resolution^4)
    counter = 1
    for(i in 1:resolution) {
      for(j in 1:resolution) {
        for(s in 1:resolution) {
          for(k in 1:resolution) {
            res_cov[[counter]] =  preds[preds$alpha_coef == unique(preds$alpha_coef)[j] & 
                                        preds$lambda_coef == unique(preds$lambda_coef)[i] &
                                        preds$lambda_spatial == unique(preds$lambda_spatial)[s] &
                                        preds$alpha_spatial == unique(preds$alpha_spatial)[k], ]
            res_coef[[counter]] =  preds[preds$alpha_cov == unique(preds$alpha_cov)[j] & 
                                         preds$lambda_cov == unique(preds$lambda_cov)[i] &
                                         preds$lambda_spatial == unique(preds$lambda_spatial)[s] &
                                         preds$alpha_spatial == unique(preds$alpha_spatial)[k], ]
            res_spatial[[counter]] =  preds[preds$alpha_coef == unique(preds$alpha_coef)[j] & 
                                            preds$lambda_coef == unique(preds$lambda_coef)[i] &
                                            preds$alpha_cov == unique(preds$alpha_cov)[s] & 
                                            preds$lambda_cov == unique(preds$lambda_cov)[k], ]
            counter = counter + 1
          }
        }
      }
    }
    res_coef = abind::abind(res_coef, along = 0L)
    res_coef = apply(res_coef, 2:3, mean)
    
    res_cov = abind::abind(res_cov, along = 0L)
    res_cov = apply(res_cov, 2:3, mean)
    
    res_spatial = abind::abind(res_spatial, along = 0L)
    res_spatial = apply(res_spatial, 2:3, mean)
    
    x_scaled = x[,1:6]
    x_scaled = sapply(x_scaled, function(s) zero_one(s))
  } else {
    
    d = data.frame(expand.grid(xn1, yn1, xn2, yn2))
    colnames(d) = c("alpha_cov", "lambda_cov", "alpha_coef", "lambda_coef")
    pp = mgcv::predict.gam(g, d)
    preds = cbind(pp, d)
    
    res_coef = vector("list", resolution^2)
    res_cov = vector("list", resolution^2)
    counter = 1
    for(i in 1:resolution) {
      for(j in 1:resolution) {
        res_cov[[counter]] =  preds[preds$alpha_coef == unique(preds$alpha_coef)[j] & preds$lambda_coef == unique(preds$lambda_coef)[i], ]
        res_coef[[counter]] =  preds[preds$alpha_cov == unique(preds$alpha_cov)[j] & preds$lambda_cov == unique(preds$lambda_cov)[i], ]
        
        counter = counter + 1
      }
    }
    res_coef = abind::abind(res_coef, along = 0L)
    res_coef = apply(res_coef, 2:3, mean)
    
    res_cov = abind::abind(res_cov, along = 0L)
    res_cov = apply(res_cov, 2:3, mean)
    
    x_scaled = x[,1:4]
    x_scaled = sapply(x_scaled, function(s) zero_one(s))
  
  }
  
  if(perf == "logLik") {
    maxP = which.min(x[[perf]])
    minP = which.max(x[[perf]])
    range = c(0.95*min(x$logLik), 1.05*max(x$logLik))
    cols = rev((grDevices::colorRampPalette(c("white", "#24526E"), bias = 1.5))(resolution))
  } else {
    maxP = which.max(x[[perf]])
    minP = which.min(x[[perf]])
    range = c(0.5, 1.0)
    cols = (grDevices::colorRampPalette(c("white", "#24526E"), bias = 1.5))(resolution)
  }
  

  perf_ind = which(perf == colnames(x), arr.ind = TRUE)
  
  if(spatial) {
    graphics::par(mfrow = c(1,3), mar = c(4,4,3,4), oma = c(2,2,2,2))
    new_image(res_cov[,c(1,2,3)], range = range, cols = cols)
    graphics::polygon(c(-0.037, -0.037, 1.037, 1.037, -0.037), c(-0.037, 1.037, 1.037, -0.037, -0.037), xpd = NA, lwd = 1.2)
    graphics::text(x = 0.5, y = -0.2, pos = 1, labels = "alpha", xpd = NA)
    graphics::text(x = -0.40, y = 0.5, pos = 2, labels = "lambda", srt = 90, xpd = NA)
    graphics::text(x = c(0.0, 1.0), y = c(-0.12, -0.12), labels = c("LASSO", "Ridge"), pos = 1, xpd = NA)
    graphics::points(x = x_scaled[maxP,1], y = x_scaled[maxP,4], pch = 8, cex = 1.5, col = "darkgreen")
    graphics::text(x = x_scaled[maxP,1], y = x_scaled[maxP,4]-0.01, pos = 1,labels = round(x[maxP,perf_ind], 2), col = "darkgreen", xpd = NA)
    graphics::points(x = x_scaled[minP,1], y = x_scaled[minP,4], pch = 8, cex = 1.5, col = "red")
    graphics::text(x = x_scaled[minP,1], y = x_scaled[minP,4]-0.01, pos = 1,labels = round(x[minP,perf_ind], 2), col = "red", xpd = NA)
    graphics::text(x = 0.5, y = 1.03, pos = 3, labels = "Covariance: alpha * lambda", xpd = NA)
    graphics::points(x_scaled[-c(minP, maxP),1], x_scaled[-c(minP, maxP),4], col = "#0F222E")
    
    new_image(res_coef[,c(1,4,5)], range = range, cols = cols)
    graphics::polygon(c(-0.037, -0.037, 1.037, 1.037, -0.037), c(-0.037, 1.037, 1.037, -0.037, -0.037), xpd = NA, lwd = 1.2)
    graphics::text(x = c(0.0, 1.0), y = c(-0.12, -0.12), labels = c("LASSO", "Ridge"), pos = 1, xpd = NA)
    graphics::text(x = 0.5, y = -0.2, pos = 1, labels = "alpha", xpd = NA)
    graphics::text(x = -0.40, y = 0.5, pos = 2, labels = "lambda", srt = 90, xpd = NA)
    graphics::points(x = x_scaled[maxP,2], y = x_scaled[maxP,5], pch = 8, cex = 1.5, col = "darkgreen")
    graphics::text(x = x_scaled[maxP,2], y = x_scaled[maxP,5]-0.01, pos = 1,labels = round(x[maxP,perf_ind], 2), xpd = NA, col = "darkgreen")
    graphics::points(x = x_scaled[minP,2], y = x_scaled[minP,5], pch = 8, cex = 1.5, col = "red")
    graphics::text(x = x_scaled[minP,2], y = x_scaled[minP,5]-0.01, pos = 1,labels = round(x[minP,perf_ind], 2), xpd = NA, col = "red")
    graphics::text(x = 0.5, y = 1.03, pos = 3, labels = "Environmental: alpha * lambda", xpd = NA)
    graphics::points(x_scaled[-c(minP, maxP),1], x_scaled[-c(minP, maxP),5], col = "#0F222E")
    
    new_image(res_spatial[,c(1,6,7)], range = range, cols = cols)
    graphics::polygon(c(-0.037, -0.037, 1.037, 1.037, -0.037), c(-0.037, 1.037, 1.037, -0.037, -0.037), xpd = NA, lwd = 1.2)
    graphics::text(x = c(0.0, 1.0), y = c(-0.12, -0.12), labels = c("LASSO", "Ridge"), pos = 1, xpd = NA)
    graphics::text(x = 0.5, y = -0.2, pos = 1, labels = "alpha", xpd = NA)
    graphics::text(x = -0.40, y = 0.5, pos = 2, labels = "lambda", srt = 90, xpd = NA)
    graphics::points(x = x_scaled[maxP,2], y = x_scaled[maxP,6], pch = 8, cex = 1.5, col = "darkgreen")
    graphics::text(x = x_scaled[maxP,2], y = x_scaled[maxP,6]-0.01, pos = 1,labels = round(x[maxP,perf_ind], 2), xpd = NA, col = "darkgreen")
    graphics::points(x = x_scaled[minP,2], y = x_scaled[minP,6], pch = 8, cex = 1.5, col = "red")
    graphics::text(x = x_scaled[minP,2], y = x_scaled[minP,6]-0.01, pos = 1,labels = round(x[minP,perf_ind], 2), xpd = NA, col = "red")
    graphics::text(x = 0.5, y = 1.03, pos = 3, labels = "Spatial: alpha * lambda", xpd = NA)
    graphics::points(x_scaled[-c(minP, maxP),1], x_scaled[-c(minP, maxP),6], col = "#0F222E")
    
    
    
    yy = rev(seq(0.35, 0.66, length.out = length(cols)+1))+0.3
    for(i in 1:length(cols)) graphics::rect(xleft = 1.22+0.0, xright = 1.26+0.0, ybottom = yy[i], ytop = yy[i+1], border = NA, xpd = NA, col = rev(cols)[i])
    graphics::text(x =1.24+0.0, y = min(yy), pos = 1, label = round(range[1], 2), xpd = NA)
    graphics::text(x =1.24+0.0, y = max(yy), pos = 3, label = round(range[2], 2), xpd = NA)
    if(perf != "logLik") label = "AUC"
    else label = "logLik"
    graphics::text(y = 1.05*mean(yy), x = 1.24+0.03, labels = label, srt = -90, pos = 4, xpd = NA)
    graphics::points(x = c(1.15, 1.15), y = c(0.2, 0.27), xpd = NA, pch = 8, col = c("darkgreen", "red"))
    if(perf == "logLik") {
      graphics::text(x = c(1.15, 1.15), y = c(0.2, 0.27), xpd = NA, label= c("lowest", "highest"), pos = 4)
    } else {
      graphics::text(x = c(1.15, 1.15), y = c(0.2, 0.27), xpd = NA, label= c("highest", "lowest"), pos = 4)
    }
    
    return(invisible(c(lambda_cov = x[maxP,]$lambda_cov,
                       alpha_cov = x[maxP,]$alpha_cov,
                       lambda_coef =  x[maxP,]$lambda_coef,
                       alpha_coef = x[maxP,]$alpha_coef,
                       lambda_spatial =  x[maxP,]$lambda_spatial,
                       alpha_spatial = x[maxP,]$alpha_spatial)))
    
  } else {
    
    graphics::par(mfrow = c(1,2), mar = c(4,4,3,4), oma = c(2,2,2,2))
    new_image(res_cov[,c(1,2,3)], range = range, cols = cols)
    graphics::polygon(c(-0.037, -0.037, 1.037, 1.037, -0.037), c(-0.037, 1.037, 1.037, -0.037, -0.037), xpd = NA, lwd = 1.2)
    graphics::text(x = 0.5, y = -0.2, pos = 1, labels = "alpha", xpd = NA)
    graphics::text(x = -0.40, y = 0.5, pos = 2, labels = "lambda", srt = 90, xpd = NA)
    graphics::text(x = c(0.0, 1.0), y = c(-0.12, -0.12), labels = c("LASSO", "Ridge"), pos = 1, xpd = NA)
    graphics::points(x = x_scaled[maxP,1], y = x_scaled[maxP,3], pch = 8, cex = 1.5, col = "darkgreen")
    graphics::text(x = x_scaled[maxP,1], y = x_scaled[maxP,3]-0.01, pos = 1,labels = round(x[maxP,perf_ind], 2), col = "darkgreen", xpd = NA)
    graphics::points(x = x_scaled[minP,1], y = x_scaled[minP,3], pch = 8, cex = 1.5, col = "red")
    graphics::text(x = x_scaled[minP,1], y = x_scaled[minP,3]-0.01, pos = 1,labels = round(x[minP,perf_ind], 2), col = "red", xpd = NA)
    graphics::text(x = 0.5, y = 1.03, pos = 3, labels = "Covariance: alpha * lambda", xpd = NA)
    graphics::points(x_scaled[-c(minP, maxP),1], x_scaled[-c(minP, maxP),3], col = "#0F222E")
    
    new_image(res_coef[,c(1,4,5)], range = range, cols = cols)
    graphics::polygon(c(-0.037, -0.037, 1.037, 1.037, -0.037), c(-0.037, 1.037, 1.037, -0.037, -0.037), xpd = NA, lwd = 1.2)
    graphics::text(x = c(0.0, 1.0), y = c(-0.12, -0.12), labels = c("LASSO", "Ridge"), pos = 1, xpd = NA)
    graphics::text(x = 0.5, y = -0.2, pos = 1, labels = "alpha", xpd = NA)
    graphics::text(x = -0.40, y = 0.5, pos = 2, labels = "lambda", srt = 90, xpd = NA)
    graphics::points(x = x_scaled[maxP,2], y = x_scaled[maxP,4], pch = 8, cex = 1.5, col = "darkgreen")
    graphics::text(x = x_scaled[maxP,2], y = x_scaled[maxP,4]-0.01, pos = 1,labels = round(x[maxP,perf_ind], 2), xpd = NA, col = "darkgreen")
    graphics::points(x = x_scaled[minP,2], y = x_scaled[minP,4], pch = 8, cex = 1.5, col = "red")
    graphics::text(x = x_scaled[minP,2], y = x_scaled[minP,4]-0.01, pos = 1,labels = round(x[minP,perf_ind], 2), xpd = NA, col = "red")
    graphics::text(x = 0.5, y = 1.03, pos = 3, labels = "Coefficients: alpha * lambda", xpd = NA)
    graphics::points(x_scaled[-c(minP, maxP),1], x_scaled[-c(minP, maxP),3], col = "#0F222E")
    yy = rev(seq(0.35, 0.66, length.out = length(cols)+1))+0.3
    for(i in 1:length(cols)) graphics::rect(xleft = 1.22+0.0, xright = 1.26+0.0, ybottom = yy[i], ytop = yy[i+1], border = NA, xpd = NA, col = rev(cols)[i])
    graphics::text(x =1.24+0.0, y = min(yy), pos = 1, label = round(range[1], 2), xpd = NA)
    graphics::text(x =1.24+0.0, y = max(yy), pos = 3, label = round(range[2], 2), xpd = NA)
    if(perf != "logLik") label = "AUC"
    else label = "logLik"
    graphics::text(y = 1.05*mean(yy), x = 1.24+0.03, labels = label, srt = -90, pos = 4, xpd = NA)
    graphics::points(x = c(1.15, 1.15), y = c(0.2, 0.27), xpd = NA, pch = 8, col = c("darkgreen", "red"))
    if(perf == "logLik") {
      graphics::text(x = c(1.15, 1.15), y = c(0.2, 0.27), xpd = NA, label= c("lowest", "highest"), pos = 4)
    } else {
      graphics::text(x = c(1.15, 1.15), y = c(0.2, 0.27), xpd = NA, label= c("highest", "lowest"), pos = 4)
    }
      
    return(invisible(c(lambda_cov = x[maxP,]$lambda_cov,
                       alpha_cov = x[maxP,]$alpha_cov,
                       lambda_coef =  x[maxP,]$lambda_coef,
                       alpha_coef = x[maxP,]$alpha_coef)))
  }
}

#' new_image function
#' @param z z matrix
#' @param cols cols for gradient
#' @param range rescale to range
new_image = function(z, cols= (grDevices::colorRampPalette(c("white", "#24526E"), bias = 1.5))(10), range = c(0.5, 1.0)) {
  graphics::plot(NULL, NULL , axes = FALSE, xlab = "", ylab = "", xlim = c(0,1), ylim = c(0,1))
  x = zero_one(z[,2])
  y = zero_one(z[,3])
  zz =cut(z[,1], breaks = seq(range[1], range[2], length.out = length(cols) + 1))
  levels(zz) = cols
  zz = as.character(zz)
  dd = diff(unique(x))[1]/2
  for(i in 1:nrow(z)) {
    graphics::rect(xleft = x[i] - dd, xright = x[i] + dd, ybottom = y[i]-dd, ytop = y[i]+dd, col = zz[i], border = NA)
  }
  graphics::axis(1, at = seq(0.0,1.0, length.out = 5), labels = round(unique(z[,2]),2)[seq(1, length(unique(x)), length.out = 5)], lwd = 0.0, lwd.ticks = 1.0)
  graphics::axis(2, at = seq(0.0,1.0, length.out = 5), labels = round(unique(z[,3]),2)[seq(1, length(unique(y)), length.out = 5)], las = 2, lwd = 0.0, lwd.ticks = 1.0)
}