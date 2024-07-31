#' Anova / Variation partitioning
#' 
#' Compute variance explained by the three fractions env, space, associations
#' 
#' @param object model of object \code{\link{sjSDM}}
#' @param samples Number of Monte Carlo samples
#' @param verbose `TRUE` or `FALSE`, indicating whether progress should be printed or not
#' @param ... optional arguments which are passed to the calculation of the logLikelihood
#' 
#' @details The ANOVA function removes each of the three fractions (Environment, Space, Associations) and measures the drop in variance explained, and thus the importance of the three fractions.
#' 
#' Variance explained is measured by Deviance as well as the pseudo-R2 metrics of Nagelkerke and McFadden
#' 
#' In downstream functions such as \code{\link{plot.sjSDManova}} or \code{\link{plot.sjSDManova}} with \code{add_shared=TRUE}.
#' The anova can get unstable for many species and few occurrences/observations. We recommend using large numbers for 'samples'.
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
#' \item{model}{model}
#' 
#' Implemented S3 methods are \code{\link{print.sjSDManova}} and \code{\link{plot.sjSDManova}}
#'  
#' @seealso \code{\link{plot.sjSDManova}}, \code{\link{print.sjSDManova}},\code{\link{summary.sjSDManova}}, \code{\link{plot.sjSDMinternalStructure}}
#' 
#' @example /inst/examples/anova-example.R
#' 
#' @import stats
#' @export

anova.sjSDM = function(object, samples = 5000L, verbose = TRUE, ...) {
  out = list()
  individual = TRUE
  samples = as.integer(samples)
  object = checkModel(object)
  
  pkg.env$fa$set_seed(object$seed)
  
  if(object$family$family$family == "gaussian") stop("gaussian not yet supported")
  
  object$settings$se = FALSE
  
  null_m = -get_null_ll(object, verbose = verbose)
  
  full_m = get_conditional_lls(object, null_m, sampling = samples, ...)
  
  ### fit different models ###
  #e_form = stats::as.formula(paste0(as.character(object$settings$env$formula), collapse = ""))
  if(!inherits(object, "spatial")) {
    out$spatial = FALSE
    
    m = update(object, env_formula = NULL, spatial_formula= ~0, biotic=bioticStruct(diag = TRUE ), verbose = verbose)
    A_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = ~0, spatial_formula= ~0, biotic=bioticStruct(diag = FALSE),  verbose = verbose)
    B_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = ~as.factor(1:nrow(object$data$X)), spatial_formula= ~0, biotic=bioticStruct(diag = FALSE ),  verbose = verbose)
    SAT_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    
    A_wo = A_m - null_m
    B_wo = B_m - null_m
    full_wo = full_m - null_m
    
    F_A = full_wo - B_wo
    F_B = full_wo - A_wo
    F_AB = full_wo - A_wo-B_wo

    anova_rows = c("Null", "F_A", "F_B", "Full")
    names(anova_rows) = c("Null", "Abiotic", "Assocations", "Full")
    
    results_discard = data.frame(models = c("F_A", "F_B","F_AB","Full", "Saturated", "Null"),
                         ll = -c(sum(null_m) + sum(F_A), sum(null_m)  + sum(F_B), sum(null_m)  + sum(F_AB), sum(full_m), sum(SAT_m), sum(null_m)))
    F_AA = F_A + F_AB*abs(F_A)/(abs(F_A)+abs(F_B))
    F_BB = F_B + F_AB*abs(F_B)/(abs(F_A)+abs(F_B))

    results_proportional = data.frame(models = c("F_A", "F_B","F_AB","Full", "Saturated", "Null"),
                                 ll = -c(sum(null_m) + sum(F_AA), sum(null_m)  + sum(F_BB), sum(null_m)  + sum(F_AB), sum(full_m), sum(SAT_m), sum(null_m)))
    
    F_AA = F_A + F_AB*0.5
    F_BB = F_B + F_AB*0.5
    
    results_equal = data.frame(models = c("F_A", "F_B","F_AB","Full", "Saturated", "Null"),
                                 ll = -c(sum(null_m) + sum(F_AA), sum(null_m)  + sum(F_BB), sum(null_m)  + sum(F_AB), sum(full_m), sum(SAT_m), sum(null_m)))
    
    
    results_ind = list("F_A"=-(null_m + F_A), "F_B"=-(null_m +F_B), "F_AB"=-(null_m + F_AB), "A" = -A_m, "B" = -B_m, "Full"=-full_m, "Saturated"=-SAT_m, "Null"=-null_m)
    
  } else {
    out$spatial = TRUE
    
    s_form = stats::as.formula(paste0(as.character(object$settings$spatial$formula), collapse = ""))
    m = update(object, env_formula = NULL, spatial_formula= ~0, biotic=bioticStruct(diag = TRUE ),  verbose = verbose)
    A_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = ~0, spatial_formula= ~0, biotic=bioticStruct(diag = FALSE), verbose = verbose)
    B_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = ~1, spatial_formula= NULL, biotic=bioticStruct(diag = TRUE), verbose = verbose)
    S_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = NULL, spatial_formula= ~0, biotic=bioticStruct(diag = FALSE ), verbose = verbose)
    AB_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = NULL, spatial_formula= NULL, biotic=bioticStruct(diag = TRUE ), verbose = verbose)
    AS_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = ~1, spatial_formula= NULL, biotic=bioticStruct(diag = FALSE ), verbose = verbose)
    BS_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    m = update(object, env_formula = ~as.factor(1:nrow(object$data$X)), spatial_formula= ~0, biotic=bioticStruct(diag = FALSE ), verbose = verbose)
    SAT_m = get_conditional_lls(m, null_m, sampling = samples, ...)
    
    
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
    
    ## discard
    
    results_discard = data.frame(models = c("F_A", "F_B","F_S","F_AB","F_AS", "F_BS", "F_ABS", "Full", "Saturated", "Null"),
                         ll = -c(sum(null_m) + sum(F_A), sum(null_m) + sum(F_B),sum(null_m) + sum(F_S), 
                                 sum(null_m) + sum(F_AB), sum(null_m) + sum(F_AS), sum(null_m) + sum(F_BS), sum(null_m) + sum(F_ABS), 
                                 sum(null_m) + sum(full_wo), sum(SAT_m), sum(null_m)))
    ## proportional
    F_AA = F_A + F_AB*abs(F_A)/(abs(F_A)+abs(F_B)) + F_AS*abs(F_A)/(abs(F_S)+abs(F_A))+ F_ABS*abs(F_A)/(abs(F_A)+abs(F_B)+abs(F_S))
    F_BB = F_B + F_AB*abs(F_B)/(abs(F_A)+abs(F_B)) + F_BS*abs(F_B)/(abs(F_S)+abs(F_B))+ F_ABS*abs(F_B)/(abs(F_A)+abs(F_B)+abs(F_S))
    F_SS = F_S + F_AS*abs(F_S)/(abs(F_S)+abs(F_A)) + F_BS*abs(F_S)/(abs(F_S)+abs(F_B))+ F_ABS*abs(F_S)/(abs(F_A)+abs(F_B)+abs(F_S))
    results_proportional = data.frame(models = c("F_A", "F_B","F_S","F_AB","F_AS", "F_BS", "F_ABS", "Full", "Saturated", "Null"),
                                     ll = -c(sum(null_m) + sum(F_AA, na.rm = TRUE), sum(null_m) + sum(F_BB, na.rm = TRUE),sum(null_m) + sum(F_SS, na.rm = TRUE), 
                                           sum(null_m) + sum(F_AB), sum(null_m) + sum(F_AS), sum(null_m) + sum(F_BS), sum(null_m) + sum(F_ABS), 
                                          sum(null_m) + sum(full_wo), sum(SAT_m), sum(null_m)))


    
    ## equal
    F_AA = F_A + F_AB*0.3333333 + F_AS*0.3333333+ F_ABS*0.3333333
    F_BB = F_B + F_AB*0.3333333 + F_BS*0.3333333+ F_ABS*0.3333333
    F_SS = F_S + F_AB*0.3333333 + F_BS*0.3333333+ F_ABS*0.3333333
    
    results_equal = data.frame(models = c("F_A", "F_B","F_S","F_AB","F_AS", "F_BS", "F_ABS", "Full", "Saturated", "Null"),
                                      ll = -c(sum(null_m) + sum(F_AA), sum(null_m) + sum(F_BB),sum(null_m) + sum(F_SS), 
                                              sum(null_m) + sum(F_AB), sum(null_m) + sum(F_AS), sum(null_m) + sum(F_BS), sum(null_m) + sum(F_ABS), 
                                              sum(null_m) + sum(full_wo), sum(SAT_m), sum(null_m)))
    
    results_ind = list("F_A"=-(null_m + F_A), "F_B"=-(null_m + F_B),"F_S"=-(null_m +F_S), "F_AB"=-(null_m +F_AB),"F_AS"=-(null_m + F_AS), 
                       "F_BS"=-(null_m + F_BS), "F_ABS"=-(null_m + F_ABS), "A" = -A_m, "B" = -B_m, "S" = -S_m, AB = AB_m, AS = AS_m, BS = BS_m,
                       "Full"=-(full_m), "Saturated"= -(SAT_m), "Null"=-null_m)
    
    anova_rows = c("Null", "F_A", "F_B", "F_S", "Full")
    names(anova_rows) = c("Null", "Abiotic", "Assocations", "Spatial", "Full")
  }
  results = 
    lapply(list(results_discard, results_proportional, results_equal), function(res) {
      
      res$`Residual deviance` = -2*(res$ll - res$ll[which(res$models == "Saturated", arr.ind = TRUE)])
      
      res$Deviance = res$`Residual deviance`[which(res$models == "Null", arr.ind = TRUE)] - res$`Residual deviance`
      R21 = function(a, b) return(1-exp(2/(nrow(object$data$Y))*(-a+b)))
      res$`R2 Nagelkerke` = R21(rep(-res$ll[which(res$models == "Null", arr.ind = TRUE)], length(res$ll)), - res$ll)
      R22 = function(a, b) 1 - (b/a)
      res$`R2 McFadden`= R22(rep(res$ll[which(res$models == "Null", arr.ind = TRUE)], length(res$ll)), res$ll)
      return(res)
    
    })
  
  # individual
  Residual_deviance_ind = lapply(results_ind, function(r) r - results_ind$Saturated)
  Deviance_ind = lapply(Residual_deviance_ind, function(r) Residual_deviance_ind$Null - r)

  R211 = function(a, b, n=1) return(1-exp(2/(n)*(-a+b)))   # divide by what?
  R2_Nagelkerke_ind = lapply(results_ind, function(r) R211(-colSums(results_ind$Null), -colSums(r), n=nrow(object$data$Y)))
  R2_Nagelkerke_sites = lapply(results_ind, function(r) R211(-rowSums(results_ind$Null), -rowSums(r), n=ncol(object$data$Y)))
  
  R222 = function(a, b) 1 - (b/a)
  R2_McFadden_ind = lapply(results_ind, function(r) R222(colSums(results_ind$Null), colSums(r)))
  R2_McFadden_sites = lapply(results_ind, function(r) R222(rowSums(results_ind$Null), rowSums(r)))
  
  R2_McFadden_ind_shared = get_shared_anova(R2_McFadden_ind)
  R2_McFadden_sites_shared = get_shared_anova(R2_McFadden_sites)
  R2_Nagelkerke_ind_shared = get_shared_anova(R2_Nagelkerke_ind)
  R2_Nagelkerke_sites_shared = get_shared_anova(R2_Nagelkerke_sites)
  
  #R2_McFadden_ind$Full = correct_R2(R2_McFadden_ind$Full)
  #R2_McFadden_sites$Full = correct_R2(R2_McFadden_sites$Full)
  
  # precalculates reduced ANOVA tables
  calculateResults <- function(res) {
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
  } else {
    printFull = results[[1]][1:4,c(1, 4, 3,5,6)]
    rownames(printFull) = printFull$models
    printFull = printFull[,-1]
    rownames(printFull) = c("Abiotic", "Associations", "Shared Abiotic+Associations", "Full")
  }
  
  toPrint = list(all = printFull,
       discard = calculateResults(results[[1]]), 
       proportional = calculateResults(results[[2]]), 
       equal = calculateResults(results[[3]])) 
  
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

get_conditional_lls = function(m, null_m, ...) {
  joint_ll = rowSums( logLik(m, individual = TRUE, ...)[[1]] )
 
   raw_ll = 
    sapply(1:ncol(m$data$Y), function(i) {
      
      reticulate::py_to_r(
        pkg.env$fa$MVP_logLik(m$data$Y[,-i], 
                              predict(m, type = "raw")[,-i], 
                              reticulate::py_to_r(m$model$get_sigma)[-i,],
                              device = m$model$device,
                              individual = TRUE,
                              dtype = m$model$dtype,
                              batch_size = as.integer(m$settings$step_size),
                              alpha = m$model$alpha,
                              link = m$family$link,
                              theta = m$theta[-i],
                              ...
                              )
        ) 
    })
  raw_conditional_ll = -( (-joint_ll) - (-raw_ll ))
  diff_ll = colSums(null_m - raw_conditional_ll)
  rates = diff_ll/sum(diff_ll)
  rescaled_conditional_lls = null_m - matrix(rates, nrow = nrow(m$data$Y), ncol = ncol(m$data$Y), byrow = TRUE) * (rowSums(null_m)-joint_ll)
  
  ### Maximal/Minimal 0?
  #rescaled_conditional_lls[rescaled_conditional_lls<0] = 0 does not work!
  
  return(rescaled_conditional_lls)
}

get_shared_anova = function(R2objt, spatial = TRUE) {
  if(spatial) {
    F_A <- R2objt$Full - R2objt$F_BS
    F_B <- R2objt$Full - R2objt$F_AB
    F_S <- R2objt$Full - R2objt$F_AS
    F_BS <- R2objt$Full - R2objt$F_A - (F_B + F_S)
    F_AB <- R2objt$Full - R2objt$F_S - (F_A + F_B)
    F_AS <- R2objt$Full - R2objt$F_B - (F_A + F_S)
    F_ABS <- R2objt$Full - (F_A + F_B + F_S + F_BS + F_AB + F_AS)
    A = F_A + F_AB*abs(F_A)/(abs(F_A)+abs(F_B)) + F_AS*abs(F_A)/(abs(F_S)+abs(F_A))+ F_ABS*abs(F_A)/(abs(F_A)+abs(F_B)+abs(F_S))
    B = F_B + F_AB*abs(F_B)/(abs(F_A)+abs(F_B)) + F_BS*abs(F_B)/(abs(F_S)+abs(F_B))+ F_ABS*abs(F_B)/(abs(F_A)+abs(F_B)+abs(F_S))
    S = F_S + F_AS*abs(F_S)/(abs(F_S)+abs(F_A)) + F_BS*abs(F_S)/(abs(F_S)+abs(F_B))+ F_ABS*abs(F_S)/(abs(F_A)+abs(F_B)+abs(F_S))
    # R2 = correct_R2(R2objt$Full) TODO Check that this can be gone
    proportional = list(F_A = A, F_B = B, F_S = S, R2 = R2objt$Full)
    A = F_A + F_AB*0.3333333 + F_AS*0.3333333+ F_ABS*0.3333333
    B = F_B + F_AB*0.3333333 + F_BS*0.3333333+ F_ABS*0.3333333
    S = F_S + F_AB*0.3333333 + F_BS*0.3333333+ F_ABS*0.3333333
    equal = list(F_A = A, F_B = B, F_S = S, R2 = R2objt$Full)
  } else {
    F_A = R2objt$Full-R2objt$B
    F_B =  R2objt$Full-R2objt$A
    F_AB = R2objt$Full - F_A -F_B
    A = F_A + F_AB*abs(F_A)/(abs(F_A)+abs(F_B))
    B = F_B + F_AB*abs(F_B)/(abs(F_A)+abs(F_B))
    S = 0
    proportional = list(F_A = A, F_B = B, F_S = S, R2 = R2objt$Full)
    A = F_A + F_AB*0.5
    B = F_B + F_AB*0.5
    S = 0
    equal = list(F_A = A, F_B = B, F_S = S, R2 = R2objt$Full)
  }
  return(list(proportional = proportional, equal = equal))
}

get_null_ll = function(object, verbose = TRUE, ...) {
  
  object_tmp = object
  object_tmp$settings$se = FALSE
  
  if(inherits(object, "spatial ")) null_pred = predict(update(object_tmp, env_formula = ~1, spatial_formula = ~0, biotic = bioticStruct(diag = TRUE), verbose = verbose))
  else null_pred = predict(update(object_tmp, env_formula = ~1, biotic = bioticStruct(diag = TRUE), verbose = verbose))
  
  if(object$family$family$family == "binomial") {
    null_m = stats::dbinom( object$data$Y, 1, null_pred, log = TRUE)
  } else if(object$family$family$family == "poisson") {
    null_m = stats::dpois( object$data$Y, null_pred, log = TRUE)
  } else if(object$family$family$family == "nbinom") {
    check_module()
    torch = pkg.env$torch
    theta = object$theta
    theta = 1.0/(softplus(theta)+0.0001)
    theta = matrix(theta, nrow = nrow(null_pred), ncol = ncol(null_pred), byrow = TRUE)
    probs = (1.0 - theta/(theta+null_pred))+0.0001 
    probs = ifelse(probs < 0.0, 0.0, probs)
    probs = ifelse(probs > 1.0, 1.0-0.0001, probs )
    theta = torch$tensor(theta, dtype = torch$float32)
    probs = torch$tensor(probs, dtype = torch$float32)
    YT = torch$tensor(object$data$Y, dtype = torch$float32)
    null_m = force_r(torch$distributions$NegativeBinomial(total_count=theta, probs=probs)$log_prob(YT)$cpu()$data$numpy())
  } else if(object$family$family$family == "gaussian") {
    warning("family = gaussian() is not fully supported yet.")
    null_m = stats::dnorm(object$data$Y, mean = null_pred, log = TRUE)
  }
    
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
#' There are four options to handle shared ANOVA fractions, which is variance that can be explained, typically as a result of collinearity, by several of the fractions:
#' 
#' 1. "all" returns the shared fractions explicitly
#' 2. "discard" discards the fractions, as typically in a type II Anova
#' 3. "proportional" distributes shared fractions proportional to the unique fractions
#' 4. "equal" distributions shared fractions equally to the unique fractions
#' 
#' @return The matrix that is printed out is silently returned
#' 
#' @example /inst/examples/anova-example.R
#' @export
summary.sjSDManova = function(object, 
                              method = c("ANOVA"),
                              fractions = c("all","discard", "proportional", "equal"), ...) {
  cat("Analysis of Deviance Table\n\n")
  method = match.arg(method)
  fractions = match.arg(fractions)
  
  out = object$to_print[[fractions]]
  stats::printCoefmat(out)
  return(invisible(out))
  
}


#' Print sjSDM anova object
#' 
#' This is a wrapper for \code{\link{summary.sjSDManova}}, maintained for backwards compatibility - prefer to use summary() instead
#'
#' @param x an object of type sjSDManova created by \code{\link{anova.sjSDM}}
#' @param ... additional arguments to \code{\link{summary.sjSDManova}}
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
#' @param ... Additional arguments to pass to \code{plot()}
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





