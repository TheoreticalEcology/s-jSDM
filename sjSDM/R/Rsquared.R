#' R-squared
#' 
#' calculate R-squared following McFadden or Nagelkerke
#' @param model model
#' @param method McFadden or Nagelkerke
#' 
#' @details
#' \loadmathjax 
#' 
#' Calculate R-squared following Nagelkerke or McFadden:
#' 
#' \itemize{
#' \item Nagelkerke: \mjseqn{R^2 = 1 - \exp(2/N \cdot (log\mathcal{L}_0 - log\mathcal{L}_1 ) )}
#' \item McFadden: \mjseqn{R^2 = 1 - log\mathcal{L}_1 / log\mathcal{L}_0  }
#'} 
#'
#' @return 
#' 
#' R-squared as numeric value
#' 
#' @author Maximilian Pichler
#' @import stats
#' @export

Rsquared = function(model, method = c("McFadden","Nagelkerke")) {
  
  method = match.arg(method)
  
  N0 = sum(get_null_ll(model))
  
  N1 = -sum(logLik(model, individual=TRUE )[[1]])
  
  if(method == "McFadden") {
    R2 = 1 - (N1/N0)
  } else {
    R2 = 1-exp(2/(nrow(model$data$Y))*(-N1+N0))
  }
  print(R2)
  return(invisible(R2))
}



 