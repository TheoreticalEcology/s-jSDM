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
#' @param conda_python_version python version to be installed in the env, default = 3.8
#' @param pip use pip installer
#' @param cuda which cuda version, 9.2 and 10.2 are supported
#' @param ... arguments passed to reticulate::conda_install()
#'
#' @export
install_sjSDM = function(method = "conda",
                         conda = "auto",
                         version = c("cpu", "gpu"),
                         envname = "r-reticulate",
                         extra_packages = NULL,
                         restart_session = TRUE,
                         conda_python_version = "3.8",
                         pip = FALSE,
                         cuda = c("10.2", "9,2"), ...) {
  
  
  is_windows = function() {
    identical(.Platform$OS.type, "windows")
  }
  
  is_unix = function() {
    identical(.Platform$OS.type, "unix")
  }
  
  is_osx = function() {
    Sys.info()["sysname"] == "Darwin"
  }
  
  is_linux = function() {
    identical(tolower(Sys.info()[["sysname"]]), "linux")
  }
  version = match.arg(version)
  cuda = match.arg(cuda)
  
  
  conda = tryCatch(reticulate::conda_binary(), error = function(e) e)
  
  if(inherits(conda, "error")) {
    reticulate::install_miniconda()
  }
 # conda_binary = reticulate::conda_binary()
 # system2(conda_binary, args=paste0("remove -y --all -n r-reticulate || true"))
 
  
  
  channel = "pytorch"
  
  
  
  if(is_windows()) {
    package = list()
    package$conda =
      switch(version,
             cpu = "pytorch torchvision torchaudio cpuonly",
             gpu = "pytorch torchvision torchaudio cudatoolkit=10.2")
    if(cuda == 9.2 && version == "gpu") package$conda = "pytorch torchvision cudatoolkit=9.2 -c pytorch -c defaults -c numba/label/dev"
    
    package$pip = 
      switch(version,
             cpu = "torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html",
             gpu = "torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html")
    if(cuda == 9.2 && version == "gpu") package$conda = "torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html"
  }
  
  if(is_linux() || is_unix()) {
    package = list()
    package$conda =
      switch(version,
             cpu = "pytorch torchvision torchaudio cpuonly",
             gpu = "pytorch torchvision torchaudio cudatoolkit=10.2")
    if(cuda == 9.2 && version == "gpu") package$conda = "pytorch torchvision cudatoolkit=9.2 -c pytorch"
    
    package$pip =
      switch(version,
             cpu = "torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html",
             gpu = "torch torchvision")
    if(cuda == 9.2 && version == "gpu") package$pip = "torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html"
  } 
  if(is_osx()) {
    package = list()
    package$conda =
      switch(version,
             cpu = "pytorch torchvision torchaudio",
             gpu = "pytorch torchvision torchaudio")
    
    package$pip =
      switch(version,
             cpu = "torch torchvision torchaudio",
             gpu = "torch torchvision torchaudio")
    
    if(version == "gpu") message("PyTorch does not provide cuda binaries for macOS, installing CPU version...\n")
  }
  
  packages = strsplit(unlist(package), " ", fixed = TRUE)
  
  error = tryCatch({
    conda_path =reticulate::conda_binary()
    conda_python = reticulate::conda_python()
    system2(conda_path, args=paste0(" create -y --force -n r-reticulate"))
    system2(conda_path, args=paste0(" install -y python=", conda_python_version))
    system2(conda_path, args=paste0(" install -y -n r-reticulate ", paste(packages$conda, collapse = " "), " -c pytorch"))
    system2(conda_python, args=paste0(" -m pip install --upgrade pyro-ppl torch_optimizer"))
    
    # if (is_osx() || is_linux() || is_unix()) {
    #   
    #   if(pip) packages$conda = packages$pip
    #   
    #   if(!is_osx()) {
    #   
    #     if (method == "conda") {
    #       
    #       reticulate::conda_install(
    #         packages = packages$conda,
    #         envname = envname,
    #         conda = conda,
    #         channel = channel,
    #         python_version = conda_python_version,
    #         pip = pip,
    #         ...
    #       )
    #       reticulate::conda_install(
    #         packages = "pyro-ppl",
    #         envname = envname,
    #         conda = conda,
    #         pip = TRUE
    #       )
    #       
    #       reticulate::conda_install(
    #         packages = "torch_optimizer",
    #         envname = envname,
    #         conda = conda,
    #         pip = TRUE
    #       )
    #     } else if (method == "virtualenv" || method == "auto") {
    #       reticulate::virtualenv_install(
    #         packages = packages$pip,
    #         envname = envname,
    #         ...
    #       )
    #     }
    #   } else {
    #     conda = reticulate::conda_binary()
    #     system(paste0(conda, " create -y -n ", envname, " python=", conda_python_version))
    #     system(paste0(conda, " install -y -n ", envname," pytorch torchvision torchaudio -c pytorch"))
    #     
    #     reticulate::conda_install(
    #       packages = "pyro-ppl",
    #       envname = envname,
    #       conda = conda,
    #       pip = TRUE
    #     )
    #     
    #     reticulate::conda_install(
    #       packages = "torch_optimizer",
    #       envname = envname,
    #       conda = conda,
    #       pip = TRUE
    #     )
    #   }
    #     
    # } else if (is_windows()) {
    #   
    #   if (method == "virtualenv") {
    #     stop("Installing PyTorch into a virtualenv is not supported on Windows",
    #          call. = FALSE)
    #   } else if (method == "conda" || method == "auto") {
    #     if(pip) packages$conda = packages$pip
    #     
    #     conda_path = reticulate::conda_binary()
    #     
    #     # reticulate::conda_install(
    #     #   packages = packages$conda,
    #     #   envname = envname,
    #     #   conda = conda,
    #     #   channel = channel,
    #     #   python_version = conda_python_version,
    #     #   pip = pip,
    #     #   ...
    #     # )
    #     # reticulate::conda_install(
    #     #   packages = "pyro-ppl",
    #     #   envname = envname,
    #     #   conda = conda,
    #     #   pip = TRUE
    #     # )
    #     # 
    #     # reticulate::conda_install(
    #     #   packages = "torch_optimizer",
    #     #   envname = envname,
    #     #   conda = conda,
    #     #   pip = TRUE
    #     # )
    #     
    #   }
    #   
    # } else {
    #   stop("Unable to install PyTorch on this platform. ",
    #        "Binary installation is available for Windows, OS X, and Linux")
    # }
  }, error = function(e) e)
  
  if(!inherits(error, "error")) {
    message("\nInstallation complete.\n\n")
    
    if (restart_session && rstudioapi::hasFun("restartSession"))
      rstudioapi::restartSession()
    
    invisible(NULL)
  } else {
    cat("\nInstallation failed... Try to install manually PyTorch (install instructions: https://github.com/TheoreticalEcology/s-jSDM\n")
    cat("If the installation still fails, please report the following error on https://github.com/TheoreticalEcology/s-jSDM/issues\n")
    cat(error$message)
  }
}


#' @title install diagnostic
#' 
#' @description Print information about available conda environments, python configs, and pytorch versions. 
#' @details If the trouble shooting guide \code{\link{installation_help}} did not help with the installation, please create an issue on \href{https://github.com/TheoreticalEcology/s-jSDM/issues}{issue tracker} with the output of this function as a quote. 
#' 
#' @seealso \code{\link{installation_help}}, \code{\link{install_sjSDM}}
#' @export
install_diagnostic = function() {
  conda_envs = reticulate::conda_list()
  
  conda = reticulate::conda_binary()
  
  configs = ""
  conda_info = ""
  suppressWarnings({
    try({
      for(n in conda_envs$name) {
        configs = paste0(configs, "\n\n\nENV: ", n)
        configs = paste0(configs, "\n\n\ntorch:\n", paste0(system(paste0(conda, " list -n" ,n, " torch*"), intern = TRUE) ,collapse = "\n" ))
        configs = paste0(configs, "\n\n\nnumpy:\n", paste0(system(paste0(conda, " list -n" ,n, " numpy*"),  intern = TRUE)  ,collapse = "\n" )   )
      }
      conda_info = paste0(system(paste0(conda, " info"), intern=TRUE), collapse = "\n")
    }, silent = TRUE)})
  
  print(conda_envs)
  cat(configs)
  cat("\n\n\n")
  print(reticulate::py_config())
  cat("\n\n\n")
  cat(conda_info)
}