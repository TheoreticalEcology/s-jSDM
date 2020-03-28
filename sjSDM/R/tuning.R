#' Cross validation of elastic net tuning
#' 
#' @param X env matrix or data.frame
#' @param Y species occurrence matrix
#' @param tune tuning strategy, random or grid search
#' @param tune_steps number of tuning steps
#' @param CV n-fold cross validation
#' @param alpha_cov weighting of l1 and l2 on covariances: \eqn{(1 - \alpha) * |cov| + \alpha ||w||^2}
#' @param alpha_coef weighting of l1 and l2 on coefficients: \eqn{(1 - \alpha) * |coef| + \alpha ||coef||^2}
#' @param lambda_cov overall regularization strength on covariances
#' @param lambda_coef overall regularization strength on coefficients
#' @param n_cores number of cores for parallelization 
#' @param ... arguments passed to sjSDM, see \code{\link{sjSDM}}
#' 
#' @example /inst/examples/sjSDM_cv-example.R
#' @seealso \code{\link{plot.sjSDM_cv}}, \code{\link{print.sjSDM_cv}}, \code{\link{summary.sjSDM_cv}}
#' @export

sjSDM_cv = function(X, Y, tune = c("random", "grid"), CV = 5L, tune_steps = 20L,
                    alpha_cov = seq(0.0, 1.0, 0.1), 
                    alpha_coef = seq(0.0, 1.0, 0.1), 
                    lambda_cov = 2^seq(-10,-1, length.out = 20),
                    lambda_coef = 2^seq(-10,-0.5, length.out = 20),
                    n_cores = NULL, 
                    ...) {
  
  tune = match.arg(tune)
  indices = 1:nrow(X)
  set = cut(sample.int(nrow(X)), breaks = CV, labels = FALSE)
  test_indices = lapply(unique(set), function(s) which(set == s, arr.ind = TRUE))
  
  tune_grid = expand.grid(alpha_cov, alpha_coef, lambda_cov, lambda_coef)
  colnames(tune_grid) = c("alpha_cov", "alpha_coef", "lambda_cov", "lambda_coef")
  
  if(tune == "random") {
    tune_samples = tune_grid[sample.int(nrow(tune_grid), tune_steps),]
  } else {
    tune_samples = tune_grid
  }
  
  
  tune_func = function(t){
    a_cov = tune_samples[t,1]
    a_coef = tune_samples[t,2]
    l_cov = tune_samples[t,3]
    l_coef = tune_samples[t,4]
    # lists work better for parallel support 
    cv_step_result = vector("list", CV)
    for(i in 1:length(test_indices)) {
      ### model ###
      X_test = X[test_indices[[i]],]
      Y_test = Y[test_indices[[i]],]
      
      X_train = X[-test_indices[[i]],]
      Y_train = Y[-test_indices[[i]],]
      
      model = sjSDM(X = X_train, Y = Y_train, 
                    l1_coefs = (1-a_coef)*l_coef,
                    l2_coefs = (a_coef)*l_coef,
                    l1_cov = (1-a_cov)*l_cov,
                    l2_cov = (a_cov)*l_cov,
                    ...)
      pred_test = predict.sjSDM(model, newdata = X_test)
      pred_train = predict.sjSDM(model)
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
      if(is.data.frame(X_test)) {
        newdata = stats::model.matrix(model$formula, X_test)
      } else {
        newdata = stats::model.matrix(model$formula, data.frame(X_test))
      }
      ll_test = model$model$logLik(newdata, Y_test,batch_size = as.integer(floor(nrow(X_test)/2)))
      cov = getCov.sjSDM(model)
      cv_step_result[[i]] = list(indices = test_indices[[i]], 
                                 pars = tune_samples[t,],
                                 pred_test = pred_test,
                                 pred_train = pred_train,
                                 ll_train = ll_train,
                                 ll_test = ll_test,
                                 auc_test = auc_test,
                                 auc_train = auc_train,
                                 auc_macro_test = auc_macro_test,
                                 auc_macro_train = auc_macro_train,
                                 coef = coef.sjSDM(model),
                                 cov = cov)
    }
    return(cv_step_result)
  }
  
  
  if(is.null(n_cores)) {
    result = vector("list", nrow(tune_samples))
    for(t in 1:nrow(tune_samples)){
      result[[t]] = tune_func(t)
    }
  } else {
    cl = snow::makeCluster(n_cores)
    control = snow::clusterEvalQ(cl, {library(sjSDM)})
    snow::clusterExport(cl, list("tune_samples", "test_indices","formula", "CV", "X", "Y", "..."), envir = environment())
    result = snow::parLapply(cl, 1:nrow(tune_samples), tune_func)
    snow::stopCluster(cl)
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
  
  for(t in 1:nrow(tune_samples)) {
    for(i in 1:CV) {
      summary_results[summary_results$iter == t & summary_results$CV_set == i, 7] = result[[t]][[i]]$ll_train
      summary_results[summary_results$iter == t & summary_results$CV_set == i, 8] = result[[t]][[i]]$ll_test[[1]]
      summary_results[summary_results$iter == t & summary_results$CV_set == i, 9] = result[[t]][[i]]$auc_test
      summary_results[summary_results$iter == t & summary_results$CV_set == i, 10] = result[[t]][[i]]$auc_macro_test
      summary_results[summary_results$iter == t & summary_results$CV_set == i, 11] = result[[t]][[i]]$auc_train
      summary_results[summary_results$iter == t & summary_results$CV_set == i, 12] = result[[t]][[i]]$auc_macro_test
    }
  }
  
  res = data.frame(tune_step = 1:nrow(tune_samples), 
                   AUC_test = rep(NA,nrow(tune_samples)),
                   AUC_test_macro = rep(NA,nrow(tune_samples)),
                   logLik = rep(NA,nrow(tune_samples)))
  
  for(t in 1:nrow(tune_samples)) {
      res[t,2] = mean(summary_results[summary_results$iter == t, 9 ])
      res[t,3] = mean(summary_results[summary_results$iter == t, 10])
      res[t,4] = sum(summary_results[summary_results$iter == t, 8])
  }
  short_summary = cbind(tune_samples, res[,-1])
  short_summary$l1_cov = (1-short_summary$alpha_cov)*short_summary$lambda_cov
  short_summary$l2_cov = short_summary$alpha_cov*short_summary$lambda_cov
  short_summary$l1_coef = (1-short_summary$alpha_coef)*short_summary$lambda_coef
  short_summary$l2_coef = short_summary$alpha_coef*short_summary$lambda_coef
  
  out = list(tune_results = result, short_summary = short_summary, summary = summary_results, settings = list(tune_samples = tune_samples, CV = CV, tune = tune))
  class(out) = c("sjSDM_cv")
  return(out)
}


#' Print a fitted sjSDM_cv model
#' 
#' @param x a model fitted by \code{\link{sjSDM_cv}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
print.sjSDM_cv = function(x, ...) {
  print(x$summary)
}


#' Return summary of a fitted sjSDM_cv model
#' 
#' @param object a model fitted by \code{\link{sjSDM_cv}}
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#' @export
summary.sjSDM_cv = function(object, ...) {
  print(object$short_summary)
}


#' Plot elastic net tuning
#' 
#' @param x a model fitted by \code{\link{sjSDM_cv}}
#' @param y unused argument
#' @param perf performance measurement to plot
#' @param resolution resolution of grid
#' @param k number of knots for the gm
#' @param ... Additional arguments to pass to \code{plot()}
#' @export
plot.sjSDM_cv = function(x, y, perf = c("logLik", "AUC", "AUC_macro"), resolution = 15,k = 3, ...) {
  oldpar = par()
  on.exit(do.call(par, oldpar))
  x = x$short_summary
  perf = match.arg(perf)
  if(perf == "AUC") perf = "AUC_test"
  if(perf == "AUC_macro") perf = "AUC_test_macro"
  
  form = paste0(perf, " ~ te(alpha_cov, lambda_cov, k = ",k,") + te(alpha_coef, lambda_coef, k = ", k, ")")
  g = mgcv::gam(stats::as.formula(form), data = x)
  
  xn1 = seq(min(x$alpha_cov), max(x$alpha_cov), length.out = resolution)
  yn1 = seq(min(x$lambda_cov), max(x$lambda_cov), length.out = resolution)
  xn2 = seq(min(x$alpha_coef), max(x$alpha_coef), length.out = resolution)
  yn2 = seq(min(x$lambda_coef), max(x$lambda_coef), length.out = resolution)
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
  x_scaled = sapply(x_scaled, function(s) scales::rescale(s, c(0,1)))
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
    text(x = c(1.15, 1.15), y = c(0.2, 0.27), xpd = NA, label= c("lowest", "highest"), pos = 4)
  } else {
    text(x = c(1.15, 1.15), y = c(0.2, 0.27), xpd = NA, label= c("highest", "lowest"), pos = 4)
  }
    
  return(invisible(c(l1_cov = (1-x[maxP,]$alpha_cov)*x[maxP,]$lambda_cov, 
                    l2_cov = (x[maxP,]$alpha_cov)*x[maxP,]$lambda_cov,
                    l1_coef = (1-x[maxP,]$alpha_coef)*x[maxP,]$lambda_coef,
                    l2_coef = x[maxP,]$alpha_coef*x[maxP,]$lambda_coef)))
}

#' new_image function
#' @param z z matrix
#' @param cols cols for gradient
#' @param range rescale to range
new_image = function(z, cols= (grDevices::colorRampPalette(c("white", "#24526E"), bias = 1.5))(10), range = c(0.5, 1.0)) {
  graphics::plot(NULL, NULL , axes = FALSE, xlab = "", ylab = "", xlim = c(0,1), ylim = c(0,1))
  x = scales::rescale(z[,2], c(0,1))
  y = scales::rescale(z[,3], c(0,1))
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


