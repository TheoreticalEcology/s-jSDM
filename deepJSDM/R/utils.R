#' @export
print.deepJmodel = function(model){

}




#' @export
plot.deepJmodel = function(model, ...){
  plot(NULL, NULL, xlim = c(1,length(model$history)), ylim = c(0, max(model$history)), xlab = "Epochs", ylab = "Loss",las =1, ...)
  points(x = 1:length(model$history), y = model$history, pch = 19, cex = 0.7)
  lines(smooth.spline(1:length(model$history), model$history, spar = 0.4), col = "red", lwd = 1.2)
}



#' useGPU
#' use a specific gpu
#' @param device number
#' @export
useGPU = function(device = 0) {
  if(!.torch$cuda$is_available()) stop("Cuda/gpu not available...")
  .device <<- .torch$device(paste0("cuda:",device))
}

#' useCPU
#' use CPU
#' @export
useCPU = function(){
  .device <<- .torch$device("cpu")
}

#' gpuInfo
#' list gpu infos
#' @export
gpuInfo = function(){
  print(.torch$cuda$get_device_properties(.device))
}


