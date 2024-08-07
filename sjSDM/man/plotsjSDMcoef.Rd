% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plot.R
\name{plotsjSDMcoef}
\alias{plotsjSDMcoef}
\title{Internal coefficients plot}
\usage{
plotsjSDMcoef(object, wrap_col = NULL, group = NULL, col = NULL, slist = NULL)
}
\arguments{
\item{object}{a model fitted by \code{\link{sjSDM}}}

\item{wrap_col}{Scales argument passed to wrap_col}

\item{group}{Define the taxonomic characteristics of a species, you need to provide a dataframe with column1 named “species” and column2 named “group”, default is NULL. For example, \code{group[1,1]== "sp1"}, \code{group[1,2]== "Mammal"}.}

\item{col}{Define colors for groups, default is NULL.}

\item{slist}{Select the species you want to plot, default is all, parameter is not supported yet.}
}
\value{
ggplot2 object
}
\description{
Plotting coefficients returned by sjSDM model.
This function only for model fitted by linear, fitted by DNN is not yet supported.
}
\examples{
\dontrun{
library(sjSDM)
# simulate community:
com = simulate_SDM(env = 6L, species = 7L, sites = 100L)

# fit model:
model = sjSDM(Y = com$response,env = com$env_weights, iter = 2L, se = TRUE,
              verbose = FALSE) 

# normal plot
plot(model)

# colored by groups
species=c("sp1","sp2","sp3","sp4","sp5","sp6","sp7")
group=c("mammal","bird","fish","fish","mammal","amphibian","amphibian")
group = data.frame(species=species,group=group)

plot(model,group=group)
}
}
\author{
CAI Wang
}
