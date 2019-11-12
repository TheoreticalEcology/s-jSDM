.onLoad = function(libname, pkgname){
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
