#' missing_installation 
#' @param miss_torch torch missing, logical 
#' @param miss_sjSDM sjSDM_py missing, logical
missing_installation = function(miss_torch, miss_sjSDM) {
  if(miss_torch) miss_one = "x PyTorch not found\n"
  else miss_one = ""
  
  if(miss_sjSDM) miss_two = "x sjSDM_py not found\n"
  else miss_two = ""
  
  out = paste0(miss_one, miss_two, "Use install_sjSDM() to install PyTorch and sjSDM_py")
  packageStartupMessage(out)
}

.onLoad = function(libname, pkgname){
  if(is_torch_available()) {
    torch <<- reticulate::import("torch")

    use_cuda <<- torch$cuda$is_available()
    if(use_cuda) {
      dtype <<- torch$float32
      device <<- torch$device("cuda:0")
    } else {
      dtype <<- torch$float64
      device <<- torch$device("cpu")
    }
    miss_torch = FALSE
  } else {
    miss_torch = TRUE
  }
  
  if(is_sjSDM_py_available()) {
    fa <<- reticulate::import("sjSDM_py")
    miss_sjSDM = FALSE
  } else {
    miss_sjSDM = TRUE
  }
  if(miss_torch || miss_sjSDM) missing_installation(miss_torch, miss_sjSDM)
}

.onAttach = function(libname, pkgname){
  if(is_torch_available()) {
    torch <<- reticulate::import("torch")
    
    use_cuda <<- torch$cuda$is_available()
    if(use_cuda) {
      dtype <<- torch$float32
      device <<- torch$device("cuda:0")
    } else {
      dtype <<- torch$float64
      device <<- torch$device("cpu")
    }
    miss_torch = FALSE
  } else {
    miss_torch = TRUE
  }
  
  if(is_sjSDM_py_available()) {
    fa <<- reticulate::import("sjSDM_py")
    miss_sjSDM = FALSE
  } else {
    miss_sjSDM = TRUE
  }
  if(miss_torch || miss_sjSDM) missing_installation(miss_torch, miss_sjSDM)
}

