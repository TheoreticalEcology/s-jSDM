\dontrun{

## Conditional predictions based on focal species
com = simulate_SDM(sites = 200L)
## first 100 observations are the training data
model = sjSDM(com$response[1:100, ], com$env_weights[1:100,])
## Assume that for the other 100 observations, only the first species is missing 
## and we want to use the other 4 species to improve the predictions:
Y_focal = com$response[101:200, ]
Y_focal[,1] = NA # set to NA because occurrences are unknown

pred_conditional = predict(model, newdata = com$env_weights[101:200,], Y = Y_focal)
pred_unconditional = predict(model, newdata = com$env_weights[101:200,])[,1]

## Compare performance:
Metrics::auc(com$response[101:200, 1], pred_conditional)
Metrics::auc(com$response[101:200, 1], pred_unconditional)

## Conditional predictions are better, however, it only works if occurrences of
## other species for new sites are known!

}