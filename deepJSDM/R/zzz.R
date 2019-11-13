.onLoad = function(libname, pkgname){
  error=
    tryCatch({
  .torch <<- reticulate::import("torch")
  .use_cuda <<- .torch$cuda$is_available()
  if(.use_cuda) {
    .dtype <<- .torch$float32
    .device <<- .torch$device("cuda:0")
  } else {
    .dtype <<- .torch$float64
    .device <<- .torch$device("cpu")
  }
  }, error = function(e) e)
  if("error" %in% class(error)) cat("Pytorch not found, run install_pytorch() \n")
}

.onAttach <- function(libname, pkgname) {
  packageStartupMessage("Welcome to deepJSDM")
  .torch <<- reticulate::import("torch")
  .use_cuda <<- .torch$cuda$is_available()
  if(.use_cuda) {
    .dtype <<- .torch$float32
    .device <<- .torch$device("cuda:0")
  } else {
    .dtype <<- .torch$float64
    .device <<- .torch$device("cpu")
  }
}
