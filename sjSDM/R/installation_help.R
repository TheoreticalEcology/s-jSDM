#' @title Installation help
#' @name installation_help
#' @docType package
#' @description Trouble shooting guide for the installation of the sjSDM package
#' 
#' We provide a function \code{\link{install_sjSDM}} to install automatically 
#' all necessary python dependencies but it can fail sometimes because of 
#' individual system settings or if other python/conda installations get into 
#' the way. 
#' 
#' @section 'PyTorch' Installation - Before you start:
#' 
#' A few notes before you start with the installation (skip this point if you 
#' do not know 'conda'):
#'\itemize{
#'  \item existing 'conda' installations:
#'   make sure you have the latest conda3/miniconda3 version and 
#'   remove unnecessary 'conda' installations.
#'  \item existing 'conda'/'virtualenv' environments (skip this point if you do not know 'conda'): 
#'  we currently enforce the usage of a specific environment called 'r-sjsdm', 
#'  so if you want use a custom environment it should be named 'r-sjsdm'
#' }
#' 
#' 
#' @section Windows - automatic installation:
#' 
#' Sometimes the automatic 'miniconda' installation 
#' (via \code{\link{install_sjSDM}}) doesn't work because of white
#' spaces in the user's name. But you can easily download and install 'conda' on
#' your own:
#' 
#' Download and install the latest 
#' \href{https://www.anaconda.com/products/individual}{'conda' version}
#' 
#' Afterwards run:\cr
#' \code{install_sjSDM(version = c("gpu")) # or "cpu" if you do not have a proper gpu device }
#' 
#' Reload the package and run the example , if this doesn't work:
#' \itemize{
#' \item Restart RStudio
#' \item Install manually 'pytorch', see the following section
#' }
#' 
#'  
#' @section Windows - manual installation:
#' 
#' Download and install the latest 'conda' version:
#' \itemize{
#' \item Install the latest 
#' \href{https://www.anaconda.com/products/individual}{'conda' version}
#' \item Open the command window (cmd.exe - hit windows key + r and write cmd)
#' }
#' Run in cmd.exe:\cr
#' \preformatted{
#' $ conda create --name r-sjsdm python=3.7
#' $ conda activate r-sjsdm
#' $ conda install pytorch torchvision cpuonly -c pytorch # cpu
#' $ conda install pytorch torchvision cudatoolkit=11.3 -c pytorch #gpu
#' $ python -m pip install pyro-ppl torch_optimizer madgrad
#' }
#' 
#' Restart R, try to run the example, and if this doesn't work:
#' \itemize{
#' \item Restart RStudio
#' \item See the 'Help and bugs' section
#' }
#' 
#'    
#' @section Linux - automatic installation:
#' 
#' Run in R:\cr
#' \code{install_sjSDM(version = c("gpu")) # or "cpu" if 
#' you do not have a proper 'gpu' device }
#' 
#' Restart R try to run the example, if this doesn't work:
#' \itemize{
#' \item Restart RStudio
#' \item Install manually 'PyTorch', see the following section
#' }
#'  
#'  
#' @section Linux - manual installation:
#' 
#' We strongly advise to use a 'conda' environment but a virtual env should also 
#' work. The only requirement is that it is named 'r-sjsdm'
#' 
#' 
#' Download and install the latest 'conda' version:
#' \itemize{
#' \item Install the latest 
#' \href{https://www.anaconda.com/products/individual}{'conda' version}
#' \item Open your terminal 
#' }
#' 
#' Run in your terminal:\cr
#' \preformatted{
#' $ conda create --name r-sjsdm python=3.7
#' $ conda activate r-sjsdm
#' $ conda install pytorch torchvision cpuonly -c pytorch # cpu
#' $ conda install pytorch torchvision cudatoolkit=11.3 -c pytorch #gpu
#' $ python -m pip install pyro-ppl torch_optimizer madgrad
#' }
#' 
#' Restart R try to run the example, if this doesn't work:
#' \itemize{
#' \item Restart RStudio
#' \item See the 'Help and bugs' section
#' }
#' 
#' 
#' @section MacOS - automatic installation:
#' 
#' Run in R:\cr
#' \code{install_sjSDM(version = c("cpu"))}
#' 
#' Restart R try to run the example, if this doesn't work:
#' \itemize{
#' \item Restart RStudio
#' \item Install manually 'PyTorch', see the following section
#' }
#' 
#' 
#' @section MacOS - manual installation:
#' 
#' Download and install the latest 'conda' version:
#' \itemize{
#' \item Install the latest 
#' \href{https://www.anaconda.com/products/individual}{'conda' version}
#' \item Open your terminal 
#' }
#' 
#' Run in your terminal:\cr
#' \preformatted{
#' $ conda create --name r-sjsdm python=3.7
#' $ conda activate r-sjsdm
#' $ python -m pip install torch torchvision torchaudio 
#' $ python -m pip install pyro-ppl torch_optimizer madgrad
#' }
#' Restart R try to run the example from, if this doesn't work:
#' \itemize{
#' \item Restart RStudio
#' \item See the 'Help and bugs' section
#' }
#'
#' @section Help and bugs:
#' 
#' To report bugs or ask for help, post a 
#' \href{https://stackoverflow.com/questions/5963269/how-to-make-a-great-r-reproducible-example}{reproducible example} 
#' via the sjSDM \href{https://github.com/TheoreticalEcology/s-jSDM/issues}{issue tracker} 
#' with a copy of the \code{\link{install_diagnostic}} output as a quote. 
NULL