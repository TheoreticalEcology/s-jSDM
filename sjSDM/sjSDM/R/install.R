#' Install sjSDM and its dependencies
#'
#' @param method installation method, auto = automatically best (conda or virtualenv), or force conda with method = "conda" or virtualenv with method = "virtualenv"
#' @param conda path to conda
#' @param version version = "default" for CPU version, or "gpu" for gpu version. (note MacOS users have to install cuda binaries by themselves)
#' @param envname Name of python env, "r-pytorch" is default
#' @param extra_packages Additional Python packages to install along with
#'   PyTorch
#' @param restart_session Restart R session after installing (note this will
#'   only occur within RStudio).
#' @param conda_python_version python version to be installed in the env, default = 3.7
#' @param cuda which cuda version, 9.2 and 10.1 are supported
#'
#'
#' @export
install_sjSDM = function(method = c("auto", "virtualenv", "conda"),
                           conda = "auto",
                           version = "default",
                           envname = "r-sjSDM",
                           extra_packages = NULL,
                           restart_session = TRUE,
                           conda_python_version = "3.7",
                           cuda = "10.1") {

  stopifnot(
    version %in% c("default", "gpu")
  )

  OS = Sys.info()['sysname']
  if(stringr::str_detect(stringr::str_to_lower(OS), "windows")) {
    package = list()
    package$conda =
      switch(version,
             default = "pytorch torchvision cpuonly -c pytorch",
             gpu = "pytorch torchvision cudatoolkit=10.1 -c pytorch")
    if(cuda == 9.2 && version == "gpu") package$conda = "pytorch torchvision cudatoolkit=9.2 -c pytorch -c defaults -c numba/label/dev"
    
    package$pip = 
      switch(version,
             default = "torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",
             gpu = "torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html")
    if(cuda == 9.2 && version == "gpu") package$conda = "torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html"
    
  } else if(stringr::str_detect(stringr::str_to_lower(OS), "linux")) {
    package = list()
    package$conda =
      switch(version,
             default = "pytorch torchvision cpuonly -c pytorch",
             gpu = "pytorch torchvision cudatoolkit=10.1 -c pytorch")
    if(cuda == 9.2 && version == "gpu") package$conda = "pytorch torchvision cudatoolkit=9.2 -c pytorch"

    package$pip =
      switch(version,
             default = "torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",
             gpu = "torch torchvision")
    if(cuda == 9.2 && version == "gpu") package$pip = "torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html"
  } else {
    package = list()
    package$conda =
      switch(version,
             default = "torch torchvision")

    package$pip =
      switch(version,
             default = "pytorch torchvision -c pytorch")
  }
  ### pytorch  Windows ###
  # pip cpu:
  # pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  # pip cuda 10.1
  # pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
  # pip cuda 9.2
  # pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
  # conda cpu:
  # conda install pytorch torchvision cpuonly -c pytorch
  # conda cuda 10.1
  # conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
  # conda cuda 9.2
  # conda install pytorch torchvision cudatoolkit=9.2 -c pytorch -c defaults -c numba/label/dev
  ### pytorch  linux ###
  # pip cpu:
  # pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  # pip cuda 10.1
  # pip install torch torchvision
  # pip cuda 9.2
  # pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
  # conda cpu:
  # conda install pytorch torchvision cpuonly -c pytorch
  # conda cuda 10.1
  # conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
  # conda cuda 9.2
  # conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
  ### pytorch  macOS ###
  # pip cpu:
  #pip install torch torchvision
  # conda
  # conda install pytorch torchvision -c pytorch

  extra_packages = unique(extra_packages)
  packages = c(package$pip, list(extra = extra_packages))
  
  reticulate::py_install(c(packages, "sjSDM_py"), envname = envname, methhod = method, conda = conda, pip = TRUE)


  # method = py_install_method_detect(envname = envname, conda = conda)
  # 
  # if((method == "virtualenv") && stringr::str_detect(stringr::str_to_lower(OS), "windows")) stop("Using virtualenv with windows is not supported", call. = FALSE)
  # switch(method,
  #        virtualenv = reticulate::virtualenv_install(envname = envname, packages = c("sjSDM_py", unlist(packages$extra), strsplit(unlist(packages$pip), " ", fixed = TRUE)[[1]])),
  #        conda = {
  #          reticulate::conda_install(envname, packages = c(unlist(packages$extra), strsplit(unlist(packages$conda), " ", fixed = TRUE)[[1]]), conda = conda, python_version = python_version)
  #          reticulate::conda_install(envname, packages = "sjSDM_py", pip = TRUE)
  #          }, stop("method is not supported"))

  if (restart_session && rstudioapi::hasFun("restartSession")) rstudioapi::restartSession()

  cat("\n Installation was successful")

  return(invisible(NULL))
}

#' is_windows
#' check if os == windows
is_windows = function() {
  OS = Sys.info()['sysname']
  return(stringr::str_detect(stringr::str_to_lower(OS), "windows"))
}


# python_has_modules
python_has_modules <- function(python, modules) {
  
  # write code to tempfile
  file <- tempfile("reticulate-python-", fileext = ".py")
  code <- paste("import", modules)
  writeLines(code, con = file)
  on.exit(unlink(file), add = TRUE)
  
  # invoke Python
  status <- system2(python, shQuote(file), stdout = FALSE, stderr = FALSE)
  status == 0L
  
}


