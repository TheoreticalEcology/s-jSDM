context("python core")

source("utils.R")

testthat::test_that("sjSDM python core", {
  skip_if_no_torch()
  
  library(sjSDM)
  path = system.file("python", package = "sjSDM")
  result = system(paste0("pytest ", path, "/"),intern = TRUE)
  testthat::expect(!grepl("failed", result[length(result)], ignore.case = TRUE), failure_message = result)
})
