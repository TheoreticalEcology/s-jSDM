
# Following https://stackoverflow.com/questions/12598242/global-variables-in-packages-in-r
# Inspired by model.gluontimets (see https://github.com/business-science/modeltime.gluonts/blob/master/R/zzz.R)
# We will use an environment for global variables
pkg.env = new.env()
pkg.env$name = "r-sjsdm"
pkg.env$torch = NULL
pkg.env$fa = NULL



.onLoad = function(libname, pkgname){
  msg( text_col( cli::rule(left = "Attaching sjSDM", right = utils::packageVersion("sjSDM")) ), startup = TRUE)
  
  # causes problems on macOS systems
  if( is_osx() ) Sys.setenv( KMP_DUPLICATE_LIB_OK=TRUE ) 
  
  # load r-sjsdm environment
  success_env = try({
    envs = reticulate::conda_list()
    env_path = envs[which(envs$name %in% "r-sjsdm", arr.ind = TRUE), 2]
    reticulate::use_python(env_path, required = TRUE)
  }, silent = TRUE)
  
  # check if dependencies are installed
  check = check_installation()
  
  # load modules only if dependencies are available
  modules_available = any(check[,2] == "0")
  if(!modules_available) {
    # load torch
    pkg.env$torch = reticulate::import("torch", delay_load = TRUE, convert = FALSE )  
    
    # 'compile' and load sjSDM python package
    path = system.file("python", package = "sjSDM")
    try({
      compile = reticulate::import("compileall", delay_load = TRUE)
      tmp = compile$compile_dir(paste0(path, "/sjSDM_py"),quiet = 2L,force=TRUE)
    }, silent = TRUE)
    pkg.env$fa = reticulate::import_from_path("sjSDM_py", path, delay_load = TRUE, convert = FALSE)
    check= cbind(check, crayon::black( c(pkg.env$torch$`__version__`, rep("", 3))))
  } 
  
  check[,2] = crayon::black( rownames(check) )
  check = cbind(check, "\n")
  
  msg(paste0(apply(check, 1, function(d) paste0(d,collapse = " "))), startup = TRUE)
  
  if(modules_available) {
    msg( crayon::red( "Torch or other dependencies not found:" ), startup = TRUE)
    info = 
      c(
      "\t1. Use install_sjSDM() to install Pytorch and conda automatically \n",
      "\t2. Installation trouble shooting guide: ?installation_help \n",
      paste0("\t3. If 1) and 2) did not help, please create an issue on ", crayon::italic(crayon::blue("<https://github.com/TheoreticalEcology/s-jSDM/issues>"))," (see ?install_diagnostic) "))
    msg( info, startup = TRUE )
  }
  invisible()
}

# copied from the tidyverse package
msg <- function(..., startup = FALSE) {
  if (startup) {
    if (!isTRUE(getOption("tidyverse.quiet"))) {
      packageStartupMessage(text_col(...))
    }
  } else {
    message(text_col(...))
  }
}

# copied from the tidyverse package
text_col <- function(x) {
  # If RStudio not available, messages already printed in black
  if (!rstudioapi::isAvailable()) {
    return(x)
  }
  if (!rstudioapi::hasFun("getThemeInfo")) {
    return(x)
  }
  theme <- rstudioapi::getThemeInfo()
  
  if (isTRUE(theme$dark)) crayon::white(x) else crayon::black(x)
  
}
