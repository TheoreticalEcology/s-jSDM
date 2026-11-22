\dontrun{

## Conditional predictions based on observed species
com = simulate_SDM(sites = 200L, species = 10L)
## first 100 observations are the training data
model = sjSDM(com$response[1:100, ], com$env_weights[1:100,], iter = 50L, verbose = FALSE)
## Assume that for the other 100 observations the first three species are missing
## and we want to use the other species to improve their predictions:
Y_focal = com$response[101:200, ]
Y_focal[, 1:3] = NA # set to NA because occurrences are unknown

## all NA cells of a site are predicted together, conditional on its observed species
pred_conditional = predict(model, newdata = com$env_weights[101:200,], Y = Y_focal)
pred_unconditional = predict(model, newdata = com$env_weights[101:200,])[, 1:3]

## Compare performance for the first species:
Metrics::auc(com$response[101:200, 1], pred_conditional[, 1])
Metrics::auc(com$response[101:200, 1], pred_unconditional[, 1])

## effective sample size of the importance sampler per site, 1 means every draw counted
summary(attr(pred_conditional, "ess"))

## Leave one species out: every species conditional on all others, in one pass
pred_loo = predict(model, newdata = com$env_weights[101:200,],
                   Y = com$response[101:200, ], loo = TRUE)

## Conditional predictions are better, however, it only works if occurrences of
## other species for new sites are known!

}
