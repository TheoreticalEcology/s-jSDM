.onLoad = function(libname, pkgname){
  if(is_torch_available()) {
    torch <<- reticulate::import("torch")
    fa <<- reticulate::import("sjSDM_py")

    use_cuda <<- torch$cuda$is_available()
    if(use_cuda) {
      dtype <<- torch$float32
      device <<- torch$device("cuda:0")
    } else {
      dtype <<- torch$float64
      device <<- torch$device("cpu")
    }
  }
}

.onAttach = function(libname, pkgname) {
  if(is_torch_available()) {
    torch <<- reticulate::import("torch")
    fa <<- reticulate::import("sjSDM_py")

    use_cuda <<- torch$cuda$is_available()
    if(use_cuda) {
      dtype <<- torch$float32
      device <<- torch$device("cuda:0")
    } else {
      dtype <<- torch$float64
      device <<- torch$device("cpu")
    }
  }
}
