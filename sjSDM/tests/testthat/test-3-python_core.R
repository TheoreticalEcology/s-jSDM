source("utils.R")

testthat::test_that("sjSDM python core", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  
  library(sjSDM)
  path = system.file("python", package = "sjSDM")
  result = system(paste0("pytest ", path, "/"),intern = TRUE)
  res = result[length(result)]
  testthat::expect(!any(strsplit(res, " ", fixed = TRUE)[[1]] %in% "failed"), failure_message = result)
})


