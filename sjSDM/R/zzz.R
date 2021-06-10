check_installation = function() {
  
  torch_ = pyro_ = torch_optimizer_ = madgrad_ = c(crayon::red(cli::symbol$cross), 0)
  
  if(reticulate::py_module_available("torch")) torch_ =  c(crayon::green(cli::symbol$tick), 1)
  if(reticulate::py_module_available("pyro")) pyro_ =  c(crayon::green(cli::symbol$tick), 1)
  if(reticulate::py_module_available("torch_optimizer")) torch_optimizer_ =  c(crayon::green(cli::symbol$tick), 1)
  if(reticulate::py_module_available("madgrad")) madgrad_ =  c(crayon::green(cli::symbol$tick), 1)
  
  return(rbind("torch" = torch_,  "torch_optimizer" = torch_optimizer_, "pyro" = pyro_, "madgrad" = madgrad_))
}

.onLoad = function(libname, pkgname){
  msg( text_col( cli::rule(left = "Attaching sjSDM", right = packageVersion("sjSDM")) ), startup = TRUE)
  check = check_installation()
  
  modules_available = any(check[,2] == "0")
  if(!modules_available) {
    torch <<- reticulate::import("torch")
    
    path = system.file("python", package = "sjSDM")
    
    Sys.setenv( KMP_DUPLICATE_LIB_OK=TRUE )
    
    try({
      compile = reticulate::import("compileall")
      tmp = compile$compile_dir(paste0(path, "/sjSDM_py"),quiet = 2L,force=TRUE)
    }, silent = TRUE)
    fa <<- reticulate::import_from_path("sjSDM_py", path)
    
    check= cbind(check, crayon::black( c(torch$`__version__`, rep("", 3))) )
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
