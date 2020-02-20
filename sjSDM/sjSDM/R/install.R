#' Install sjSDM and its dependencies
#'
#' @param method installation method, auto = automatically best (conda or virtualenv), or force conda with method = "conda" or virtualenv with method = "virtualenv"
#' @param conda path to conda
#' @param version version = "cpu" for CPU version, or "gpu" for gpu version. (note MacOS users have to install cuda binaries by themselves)
#' @param envname Name of python env, "r-pytorch" is default
#' @param extra_packages Additional Python packages to install along with
#'   PyTorch
#' @param restart_session Restart R session after installing (note this will
#'   only occur within RStudio).
#' @param conda_python_version python version to be installed in the env, default = 3.6
#' @param cuda which cuda version, 9.2 and 10.1 are supported
#'
#'
#' @export
install_sjSDM = function(method = "conda",
                           conda = "auto",
                           version = c("cpu", "gpu"),
                           envname = "r-sjSDM",
                           extra_packages = NULL,
                           restart_session = TRUE,
                           conda_python_version = "3.6",
                           cuda = c("10.1", "9,2")) {

  version = match.arg(version)
  cuda = match.arg(cuda)

  OS = Sys.info()['sysname']
  if(stringr::str_detect(stringr::str_to_lower(OS), "windows")) {
    package = list()
    package$conda =
      switch(version,
             cpu = "pytorch torchvision cpuonly -c pytorch",
             gpu = "pytorch torchvision cudatoolkit=10.1 -c pytorch")
    if(cuda == 9.2 && version == "gpu") package$conda = "pytorch torchvision cudatoolkit=9.2 -c pytorch -c defaults -c numba/label/dev"
    
    package$pip = 
      switch(version,
             cpu = "torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",
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
  
  reticulate::py_install(packages = c("--user", stringr::str_split_fixed(unlist(packages), " ", n = Inf)[1,], "sjSDM_py"), envname = envname, method = method, conda = conda, pip = TRUE, python_version = conda_python_version)
  
  reticulate::use_condaenv(envname)
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


sjSDM::install_sjSDM()
