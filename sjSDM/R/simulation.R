#' Simulate joint Species Distribution Models
#'
#' @param env number of environment variables
#' @param sites number of sites
#' @param species number of species
#' @param correlation correlated species TRUE or FALSE, can be also a function or a matrix
#' @param weight_range sample true weights from uniform range, default -1,1
#' @param link probit, logit or identical
#' @param response pa (presence-absence) or count
#' @param sparse sparse rate
#' @param tolerance tolerance for sparsity check
#' @param iter tries until sparse rate is achieved
#' @param seed random seed. Default = 42
#'
#' @description Simulate species distributions
#'
#' @details Probit is not possible for abundance response (response = 'count')
#' 
#' @return
#' 
#' List of simulation results:
#' 
#' \item{env}{Number of environmental covariates}
#' \item{species}{Number of species}
#' \item{sites}{Number of sites}
#' \item{link}{Which link}
#' \item{response_type}{Which response type}
#' \item{response}{Species occurrence matrix}
#' \item{correlation}{Species covariance matirx}
#' \item{species_weights}{Species-environment coefficients}
#' \item{env_weights}{Environmental covariates}
#' \item{corr_acc}{Method to calculate sign accurracy}
#'
#' @author Maximilian Pichler
#' @export


simulate_SDM = function(
  env = 5L,
  sites = 100L,
  species = 5L,
  correlation = TRUE,
  weight_range = c(-1,1),
  link = "probit",
  response = "pa",
  sparse = NULL,
  tolerance = 0.05,
  iter = 20L,
  seed = NULL
){

  if(!is.null(seed)) set.seed(seed)

  out = list()

  if(is.null(sparse)){
      if(is.function(correlation)) species_covariance = correlation()
      
      if(is.matrix(correlation)) species_covariance = correlation
      
      if(is.logical(correlation)) {
    
        if(correlation){
            tmp = matrix(0, species, species)
            tmp[lower.tri(tmp)] = stats::runif((species* (species + 1) / 2 ) - species, -1, 1)
            diag(tmp) = 1
            # positive definite matrix
            tmp = tmp %*% t(tmp)
            species_covariance = stats::cov2cor(tmp)
        } else {
          species_covariance = diag(1, species, species)
        }
        
      }
  } else {

    for(m in 1:iter){
      sigma = diag(rep(1, species))
      target = 1 - sparse
      loss = function(param){
        k = param
        len = sum(lower.tri(sigma))
        sigma[lower.tri(sigma)] = sample(c(rep(0,ceiling((1-k)*len)), stats::rnorm(ceiling(k*len))),len)
        sigma  = sigma %*% t(sigma)
        abs(target - mean(abs(sigma[lower.tri(sigma)]) > 0.0))
      }
      pars = stats::optim(sparse, loss, lower = 0.0, upper = 1.0, method = "L-BFGS-B")
      k = pars$par
      for(i in 1:1e3){
        sigma = diag(rep(1, species))
        len = sum(lower.tri(sigma))
        sigma[lower.tri(sigma)] = sample(c(rep(0,ceiling((1-k)*len)), stats::rnorm(ceiling(k*len))),len)
        sigma  = sigma %*% t(sigma)
        species_covariance = stats::cov2cor(sigma)

        acc = abs(mean(abs(sigma[lower.tri(sigma)]) > 0.0) - target)
        if(is.na(acc)) acc = Inf
        if(acc < tolerance) break
        if(i == 1e3) cat("Iter: ",m, " Not able to find a sparse covariance matrix with the specified rate \n")
      }
      if(acc < tolerance) break
    }
  }

  species_env_weights =
    matrix(stats::runif(species * env, weight_range[1], weight_range[2]), nrow = env, ncol = species)

  env_weights =
    matrix(stats::runif(sites * env, -1, 1), nrow = sites, ncol = env)

  linear_reponse =
    env_weights %*% species_env_weights

  re_i_j =
    mvtnorm::rmvnorm(sites, mean = rep(0,species), sigma = species_covariance)

  raw_response = linear_reponse + re_i_j
  

  inv_logit = function(x) 1/(1+exp(-x))

  # logit is kind of hard coded, but I think for the time its easier then additional sampling
  site_response =
    if(link == "probit"){
      apply(raw_response,1:2, function(x) return(if(x>0) 1 else 0))
    } else if(link == "logit"){
      apply(raw_response,1:2, function(x) return(if(inv_logit(x)>0.5) 1 else 0))
    } else if(response == "count"){
      apply(raw_response,1:2, function(x) return(floor(exp(x))))
    } else if(response == "identical") {
      raw_response
    }

  out$resample = function(){
    re_i_j =
      mvtnorm::rmvnorm(sites, mean = rep(0,species), sigma = species_covariance)

    raw_response = linear_reponse + re_i_j


    site_response =
      if(link == "probit"){
        apply(raw_response,1:2, function(x) return(if(x>0) 1 else 0))
      } else if(link == "logit"){
        apply(raw_response,1:2, function(x) return(if(inv_logit(x)>0.5) 1 else 0))
      } else if(response == "count"){
        apply(raw_response,1:2, function(x) return(floor(exp(x))))
      } else if(response == "identical") {
        raw_response
      }
    return(site_response)

  }

  out$env = env
  out$species = species
  out$sites = sites
  out$link = link
  out$response_type = response
  out$response = site_response
  out$correlation = species_covariance
  out$species_weights = species_env_weights
  out$env_weights = env_weights
  out$corr_acc = function(cor){
    ind = lower.tri(species_covariance)
    true = species_covariance[ind]
    pred = cor[ind]
    d = sum((true < 0) == (pred < 0))
    return(d/sum(lower.tri(species_covariance)))
  }


  class(out) = c("jsdm", "simulation")

  return(out)
}
