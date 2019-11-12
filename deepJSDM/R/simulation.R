#' Simulate joint Species Distribution Models
#'
#' @param env number of environment variables
#' @param sites number of sites
#' @param species number of species
#' @param correlation correlated species TRUE or FALSE
#' @param weight_range sample true weights from uniform range, default -1,1
#' @param link probit, logit or idential
#' @param response pa (presence-absence) or count
#' @param seed random seed. Default = 42
#'
#' @description Simulate species distributions
#'
#' @details probit is not possible for response = count
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
  seed = 42
){
  # Check Inputs:
  require(assertthat, quietly = TRUE)
  assert_that(is.count(env),
              is.count(sites),
              is.count(species),
              is.flag(correlation),
              length(weight_range) == 2,
              sites > 0,
              env > 0,
              species > 0,
              weight_range[1] < weight_range[2],
              link %in% c("probit", "logit", "idential"),
              response %in% c("pa", "count"),
              !(link == "probit" && response == "count")
  )

  if(!is.null(seed)) set.seed(seed)

  out = list()

  species_covariance =
    if(correlation){
      tmp = matrix(0, species, species)
      tmp[lower.tri(tmp)] = runif((species* (species + 1) / 2 ) - species, -1, 1)
      diag(tmp) = 1

      # positive definite matrix
      tmp = tmp %*% t(tmp)
      tmp = cov2cor(tmp)

    } else {
      diag(1, species, species)
    }

  species_env_weights =
    matrix(runif(species * env, weight_range[1], weight_range[2]), nrow = env, ncol = species)

  env_weights =
    matrix(runif(sites * env, -1, 1), nrow = sites, ncol = env)

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
    #d = dist(t(cbind(true, pred)))
    return(d/sum(lower.tri(species_covariance)))
  }


  class(out) = c("deepJ", "simulation")

  return(out)
}
