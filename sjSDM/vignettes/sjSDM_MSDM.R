## ---- echo = F, message = F---------------------------------------------------
set.seed(123)

## ----global_options, include=FALSE--------------------------------------------
knitr::opts_chunk$set(fig.width=7, fig.height=4.5, fig.align='center', warning=FALSE, message=FALSE, cache = FALSE)

## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = ""
)

## ----eval=FALSE---------------------------------------------------------------
#  library(sjSDM)
#  Env = eucalypts$env
#  PA = eucalypts$PA
#  head(PA)

## ----  eval = TRUE,echo = FALSE, results = TRUE-------------------------------
cat("
     ALA ARE BAX CAM GON MEL OBL OVA WIL ALP VIM ARO.SAB
[1,]   0   0   0   0   0   0   0   0   1   1   0       0
[2,]   0   0   0   0   0   0   1   0   1   1   0       0
[3,]   0   0   1   0   0   0   0   0   1   1   0       0
[4,]   0   0   1   0   0   0   0   0   1   0   0       0
[5,]   0   0   1   0   0   0   1   0   0   0   0       0
[6,]   0   0   0   0   0   0   0   0   1   1   0       0"
)

## ---- eval=FALSE--------------------------------------------------------------
#  dnn = sjSDM(Y = PA,
#              env = DNN(scale(Env), hidden = c(20L, 20L)),
#              biotic = bioticStruct(diag = TRUE))

## ----  eval = TRUE,echo = FALSE, results = TRUE-------------------------------
cat("Iter: 100/100 100%|██████████| [00:03, 27.17it/s, loss=2.642]"
)

## ---- eval=FALSE--------------------------------------------------------------
#  summary(dnn)

## ----  eval = TRUE,echo = FALSE, results = TRUE-------------------------------
cat("
Family:  binomial 

LogLik:  -1166.957 
Regularization loss:  0 

Species-species correlation matrix: 

	sp1	1.0000											
	sp2	0.0000	1.0000										
	sp3	0.0000	0.0000	1.0000									
	sp4	0.0000	0.0000	0.0000	1.0000								
	sp5	0.0000	0.0000	0.0000	0.0000	1.0000							
	sp6	0.0000	0.0000	0.0000	0.0000	0.0000	1.0000						
	sp7	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	1.0000					
	sp8	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	1.0000				
	sp9	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	1.0000			
	sp10	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	1.0000		
	sp11	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	1.0000	
	sp12	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	1.0000



Env architecture:
===================================
Layer_1:	 (8, 20)
Layer_2:	 SELU
Layer_3:	 (20, 20)
Layer_4:	 SELU
Layer_5:	 (20, 12)
===================================
Weights :	 800
    "
)


## ----eval=FALSE---------------------------------------------------------------
#  jsdm_dnn = sjSDM(Y = PA,
#                   env = DNN(scale(Env), hidden = c(20L, 20L)),
#                   biotic = bioticStruct(diag = FALSE))

## ----  eval = TRUE,echo = FALSE, results = TRUE-------------------------------
cat("Iter: 100/100 100%|██████████| [00:03, 30.99it/s, loss=2.427]"
)

## ---- eval=FALSE--------------------------------------------------------------
#  summary(jsdm_dnn)

## ----  eval = TRUE,echo = FALSE, results = TRUE-------------------------------
cat("
Family:  binomial 

LogLik:  -1054 
Regularization loss:  0 

Species-species correlation matrix: 

	sp1	 1.0000											
	sp2	 0.0350	 1.0000										
	sp3	-0.1860	-0.4460	 1.0000									
	sp4	 0.0380	-0.1490	-0.3020	 1.0000								
	sp5	-0.1810	 0.4300	 0.0350	 0.0170	 1.0000							
	sp6	 0.2800	-0.3680	-0.0970	 0.2440	-0.3590	 1.0000						
	sp7	 0.0100	 0.1290	-0.0150	-0.4170	-0.4230	-0.1960	 1.0000					
	sp8	-0.0710	 0.0450	-0.1920	 0.0550	 0.1390	 0.1830	-0.2070	 1.0000				
	sp9	-0.0340	 0.5570	-0.2120	-0.2140	 0.4900	-0.3970	-0.0960	-0.0650	 1.0000			
	sp10	 0.1860	 0.0300	-0.7140	 0.5500	-0.1740	 0.3830	-0.2520	 0.2700	-0.0840	 1.0000		
	sp11	-0.2000	-0.1000	 0.0770	-0.0260	-0.2420	-0.0480	 0.4150	 0.2080	-0.4050	-0.0890	 1.0000	
	sp12	 0.1090	 0.3410	-0.3650	-0.0880	 0.0620	-0.2120	 0.0530	-0.2730	 0.4820	 0.1550	-0.4670	 1.0000



Env architecture:
===================================
Layer_1:	 (8, 20)
Layer_2:	 SELU
Layer_3:	 (20, 20)
Layer_4:	 SELU
Layer_5:	 (20, 12)
===================================
Weights :	 800    
    "
)

## ---- eval=FALSE--------------------------------------------------------------
#  association = getCor(jsdm_dnn)
#  fields::image.plot(association)

## ---- eval=FALSE--------------------------------------------------------------
#  predict_wrapper = function(model, newdata) predict(model, newdata = newdata)[,1]
#  
#  library(iml)
#  predictor =
#    Predictor$new(jsdm_dnn,
#                  data = as.data.frame(scale(Env)),
#                  y = PA[,1],# First species
#                  predict.function = predict_wrapper
#                  ) # First species

## ---- eval = FALSE------------------------------------------------------------
#  plot(FeatureEffect$new(predictor, feature = "Rockiness"))

