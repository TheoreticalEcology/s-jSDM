set.seed(77)

# Libraries ---------------------------------------------------------------
library(tidyverse)

# General Parameters ------------------------------------------------------
# To make things faster when testing different parameters, these are the ones we are playing with

disp_low <- 0.01
disp_med <- 0.05
disp_hi <- 0.1

niche_broad <- 2.0
niche_narrow <- 0.8

# Functions ---------------------------------------------------------------
# All of these scripts have functions only

# These will prepare a list of parameters in the right format
source("testingHMSC/manuscript_functions/prep_pars_fx.R")

# Original functions for the metacommunity simulation. Note addition of a quadratic response to the environment. Which is what we've used here.
source("testingHMSC/manuscript_functions/metacom_sim_fx.R")

# The actual metacommunity simulation that uses the functions above. It saves the snapshot of the metacommunity after 200 time steps.
source("testingHMSC/manuscript_functions/main_sim_fx.R")

# This full process involves running the metacommunity simulation, getting in the hmsc format, doing variation partitioning for species and sites.
source("testingHMSC/manuscript_functions/full_process_fx.R")

# Output processing involves changing structure of data from lists to dataframes and using those to get the figures
source("testingHMSC/manuscript_functions/output_processing_fx.R")

# Functions to check convergence based on previous conversations with Guillaume. Raftery and Gelman plots
source("testingHMSC/manuscript_functions/convergence_fx.R")

# Original landscape ------------------------------------------------------
# This is the original landscape provided in the Dropbox. They were previously saved as text files, it's the same data, just different format.


XY <- readRDS("testingHMSC/manuscript_functions/fixedLandscapes/orig-no-seed-XY.RDS")
E <- readRDS("testingHMSC/manuscript_functions/fixedLandscapes/orig-no-seed-E.RDS")
MEMsel <- readRDS("testingHMSC/manuscript_functions/fixedLandscapes/orig-no-seed-MEMsel.RDS")


# This first part is only setting the parameters for the different scenarios. After that, the running cycles is actually running the metacommunity simulations, fitting the model and then variation partitioning.

# A no interactions, narrow niche
scen1pars <- prep_pars(N = 1000, D = 1, R = 12, breadth = niche_narrow, nicheOpt = NULL, alpha = disp_med,
                       interx_col = 0, interx_ext = 0, makeRDS = FALSE)
# NOTE: the nicheOpt is set to null to follow the default, which gives species evenly spaced optima for the number of species we state with R. 

# B: no interactions, broad niche
scen2pars <- prep_pars(N = 1000, D = 1, R = 12, breadth = niche_broad, nicheOpt = NULL, alpha = disp_med,
                       interx_col = 0, interx_ext = 0, makeRDS = FALSE)

# C : with interactions, narrow niche

scen3pars <- prep_pars(N = 1000, D = 1, R = 12, niche_narrow, nicheOpt = NULL, alpha = disp_med,
                       interx_col = 1.0, interx_ext = 1.0, makeRDS = FALSE)

# D: with interactions, broad niche

scen4pars <- prep_pars(N = 1000, D = 1, R = 12, breadth = niche_broad, nicheOpt = NULL, alpha = disp_med,
                       interx_col = 1.0, interx_ext = 1.0, makeRDS = FALSE)

# E: half of the species with interactions, the other half without

scen5pars <- list(scen1_a = prep_pars(N = 1000, D = 1, R = 6, breadth = niche_narrow, nicheOpt = NULL, 
                                      alpha = disp_med,
                                      interx_col = 0, interx_ext = 0, makeRDS = FALSE),
                  scen1_b = prep_pars(N = 1000, D = 1, R = 6, breadth = niche_narrow, nicheOpt = NULL, 
                                      alpha = disp_med,
                                      interx_col = 1.0, interx_ext = 1.0, makeRDS = FALSE))


# F: a third of the species with low, med and high dispersal levels, without interactions.

scen6pars <- list(scen2_a = prep_pars(N = 1000, D = 1, R = 4, breadth = niche_narrow, nicheOpt = NULL, 
                                      alpha = disp_low,
                                      interx_col = 0, interx_ext = 0, makeRDS = FALSE),
                  scen2_b = prep_pars(N = 1000, D = 1, R = 4, breadth = niche_narrow, nicheOpt = NULL, 
                                      alpha = disp_med,
                                      interx_col = 0, interx_ext = 0, makeRDS = FALSE),
                  scen2_c = prep_pars(N = 1000, D = 1, R = 4, breadth = niche_narrow, nicheOpt = NULL, 
                                      alpha = disp_hi,
                                      interx_col = 0, interx_ext = 0, makeRDS = FALSE))

# G: a third of the species with low, med and high dispersal levels, with interactions.

scen7pars <- list(scen3_a = prep_pars(N = 1000, D = 1, R = 4, breadth = niche_narrow, nicheOpt = NULL, 
                                      alpha = disp_low,
                                      interx_col = 1.0, interx_ext = 1.0, makeRDS = FALSE),
                  scen3_b = prep_pars(N = 1000, D = 1, R = 4, breadth = niche_narrow, nicheOpt = NULL, 
                                      alpha = disp_med,
                                      interx_col = 1.0, interx_ext = 1.0, makeRDS = FALSE),
                  scen3_c = prep_pars(N = 1000, D = 1, R = 4, breadth = niche_narrow, nicheOpt = NULL, 
                                      alpha = disp_hi,
                                      interx_col = 1.0, interx_ext = 1.0, makeRDS = FALSE))

fig2pars <- list(scen1pars = scen1pars, scen2pars = scen2pars, 
                 scen3pars = scen3pars, scen4pars = scen4pars,
                 scen5pars = scen5pars, scen6pars = scen6pars, 
                 scen7pars = scen7pars)


# RUN THE CYCLES
sims = vector("list", 7)
for(j in 1:7){
  
  if(j < 5) {
    sims[[j]] <- metacom_sim4HMSC(XY = XY, E = E, pars = fig2pars[[j]],
                             nsteps = 200, occupancy = 0.8, niter = 5,
                             makeRDS = FALSE)
  } else {
    sims[[j]] <- metacom_sim4HMSC_multParams(XY = XY, E = E, pars = fig2pars[[j]],
                                             nsteps = 200, occupancy = 0.8, niter = 5,  makeRDS = FALSE)
  }
}

saveRDS(list(pars = fig2pars, simulation = sims), "data_process_based.RDS")
