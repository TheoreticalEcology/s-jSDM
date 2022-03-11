source("utils.R")


paths = c("../../vignettes", "../vignettes")
if((!any(sapply(paths, dir.exists)))) testthat::skip("No vignettes found...")
path = paths[sapply(paths, dir.exists)]
rmds = c(paste0(path, "/Dependencies.Rmd"),
         paste0(path, "/sjSDM.Rmd")
)

trim_name = function(name) {
  name = stringr::str_replace_all( name , "'", " ")
  name = stringr::str_replace_all( name , '"', " ")
  name = stringr::str_remove_all(name, "-")
  return(name)
}

create_test = function(input_file) {
  name = strsplit(input_file, ".", fixed = TRUE)
  name = name[[1]][length(name[[1]])-1]
  name = strsplit(name, "/", fixed = TRUE)[[1]][3]
  output = paste0(name, ".R")
  knitr::opts_chunk$set(comment = NA)
  knitr::purl(input_file, output = output, documentation = 1, quiet = TRUE)
  knitr::opts_chunk$set(comment = "##")
  input = file(output)
  code = readLines(input)
  close(input)
  file.remove(output)
  result = ''
  open = FALSE
  end = c( "list2env(as.list(environment()), envir = .GlobalEnv)", "}, NA)})" )
  
  # taken from https://stackoverflow.com/questions/6451152/how-to-catch-integer0
  is.empty <- function(x, mode = NULL){
    if (is.null(mode)) mode <- class(x)
    identical(vector(mode, 1), c(x, vector(class(x), 1)))
  }
  
  for(i in 1:length(code)){
    start = grep("## ----", code[i])
    if(is.empty(start)) start=FALSE
    if(start == 1) start = TRUE
    if(start){
      if(open){
        result = c(result, end)
        result = c(result, paste0("testthat::test_that('",name,"__", trim_name(code[i]),
                                  "', {testthat::expect_error( {"))
      }else{
        result = c(result, paste0("testthat::test_that('",name,"__", trim_name(code[i]),
                                  "', {testthat::expect_error( {"))
        open = TRUE
      }
    }
    result = c(result, code[i])
  }
  
  result = c(result, end) #, "rm(list=ls())", "gc()")
  test_file = paste0("test-", name, ".R")
  writeLines(result, con = test_file)
  return(test_file)
}

vignette_tests = sapply(rmds, create_test)
sapply(vignette_tests, source)
file.remove(vignette_tests)
