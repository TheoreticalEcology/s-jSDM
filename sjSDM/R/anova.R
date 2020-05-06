#' anova
#' 
#' anova
#' 
#' @param object model of object...
#' @export

anova.sjSDM = function(object, ...) {
  
  model$settings$env
  splits = createSplit(nrow(model$settings$env$X), 5L)
  
  if(!is.null(object$spatial_weights)){
    full_ll = logLik(model)
    full_r = Rsquared(model)
    full = list(ll = full_ll, R = full_r)
    
    .full = sapply(splits, function(sp) turnOn(model, modules = "ABS", test = sp))
    full = list(ll = sum(unlist(.full[1,])), R = mean(unlist(.full[2,])))
    
    .A = sapply(splits, function(sp) turnOn(model, modules = "A", test = sp))
    A = list(ll = sum(unlist(.A[1,])), R = mean(unlist(.A[2,])))
    
    .B = sapply(splits, function(sp) turnOn(model, modules = "B", test = sp))
    B = list(ll = sum(unlist(.B[1,])), R = mean(unlist(.B[2,])))
    
    .S = sapply(splits, function(sp) turnOn(model, modules = "S", test = sp))
    S = list(ll = sum(unlist(.S[1,])), R = mean(unlist(.S[2,])))
    
    .AB = sapply(splits, function(sp) turnOn(model, modules = "AB", test = sp))
    AB = list(ll = sum(unlist(.AB[1,])), R = mean(unlist(.AB[2,])))
    
    .AS = sapply(splits, function(sp) turnOn(model, modules = "AS", test = sp))
    AS = list(ll = sum(unlist(.AS[1,])), R = mean(unlist(.AS[2,])))
    
    .BS = sapply(splits, function(sp) turnOn(model, modules = "BS", test = sp))
    BS = list(ll = sum(unlist(.BS[1,])), R = mean(unlist(.BS[2,])))
    
    # A = turnOn(model, modules = "A")
    # B = turnOn(model, modules = "B")
    # S = turnOn(model, modules = "S")
    # AB = turnOn(model, modules = "AB")
    # AS = turnOn(model, modules = "AS")
    # BS = turnOn(model, modules = "BS")
    
    res = data.frame(matrix(NA, 9,3))
    colnames(res) = c("Group", "relativll", "relativR2")    
    res$Group = c("Empty", "A", "B", "S", "A+B", "A+S", "B+S", "A+B+S", "Full")
    res[1,2:3] = unlist(empty)
    
    res[2,2] = -empty$ll + A$ll # A
    res[3,2] = -empty$ll + B$ll # B
    res[4,2] = -empty$ll + S$ll # S
    
    res[5,2] = -empty$ll + AB$ll - (-res[2,2] + -res[3,2] )# A+B
    res[6,2] = -empty$ll + AS$ll - (-res[2,2] + -res[4,2] )# A+S
    res[7,2] = -empty$ll + BS$ll - (-res[4,2] + -res[3,2] )# B+S
    
    res[8,2] = -empty$ll + full$ll - sum(-res[2:7,2]) # A+B+S
    res[9,2] = -empty$ll + full$ll # Full
    
    res[2,3] = empty$R + A$R
    res[3,3] = empty$R + B$R
    res[4,3] = empty$R + S$R
    
    res[5,3] = empty$R + AB$R - res[2,3] - res[3,3]
    res[6,3] = empty$R + AS$R - res[2,3] - res[4,3]
    res[7,3] = empty$R + BS$R - res[4,3] - res[3,3]
    res[8,3] = empty$R + full$R - sum(res[2:7,3])
    res[9,3] = empty$R + full$R
    
    return(res)
  } else {
    full_ll = logLik(model)
    full_r = Rsquared(model)
    full = list(ll = full_ll, R = full_r)
    
    empty = turnOn(model, modules = "")
    A = turnOn(model, modules = "A")
    B = turnOn(model, modules = "B")
    AB = turnOn(model, modules = "AB")
    
    res = data.frame(matrix(NA, 5,3))
    colnames(res) = c("Group", "relativll", "relativR2")    
    res$Group = c("Empty", "A", "B", "A+B", "Full")
    res[1,2:3] = unlist(empty)
    
    res[2,2] = -empty$ll + A$ll # A
    res[3,2] = -empty$ll + B$ll # B
    
    res[4,2] = -empty$ll + AB$ll - (-res[2,2] + -res[3,2] )# A+B
    
    res[5,2] = -empty$ll + full$ll # Full
    
    res[2,3] = empty$R + A$R
    res[3,3] = empty$R + B$R
    
    res[4,3] = empty$R + AB$R - res[2,3] - res[3,3]
    res[5,3] = empty$R + full$R
    
    
    }
  return(res)
}


createSplit = function(n=NULL,CV=5) {
  set = cut(sample.int(n), breaks = CV, labels = FALSE)
  test_indices = lapply(unique(set), function(s) which(set == s, arr.ind = TRUE))
  return(test_indices)
}

# test = createSplit(100L, 5L)[[1]]
# 
# turnOn(model, modules = c("A"), test=test)


turnOn = function(model, modules = c("AB"), test= NULL, ...) {
  modules = strsplit(modules,split = "")[[1]]
  
  env = model$settings$env
  spatial = model$settings$spatial
  
  if(!is.null(test)) {
    env$X = env$X[-test,,drop=FALSE]
    if(!is.null(spatial)) {
      spatial$X = spatial$X[-test,,drop=FALSE]
    }
    Y = model$data$Y[-test,,drop=FALSE]
  } else {
    Y = model$data$Y
  }
  env2 = env
  env2$X = matrix(0.0, nrow(env2$X),ncol(env2$X))
  biotic2 = bioticStruct(diag = TRUE)
  spatial2 = spatial
  if(!is.null(spatial2)) {
    spatial2$X = matrix(0.0, nrow(spatial2$X),ncol(spatial2$X))
    test_sp = matrix(0.0, nrow(model$settings$spatial$X[test,,drop=FALSE]),ncol(spatial2$X))
  }
  test_env = matrix(0.0, nrow(model$settings$env$X[test,,drop=FALSE]),ncol(env2$X))
  

  for(i in modules){
    if(i == "A") {
      env2 = env
      test_env = model$settings$env$X[test,,drop=FALSE]
    }
    if(i == "B") biotic2 = model$settings$biotic
    if(i == "S") {
      spatial2 = spatial
      test_sp = model$settings$spatial$X[test,,drop=FALSE]
    }
  }
  
  m2 = sjSDM(Y = Y, 
             env = env2, 
             biotic = biotic2,
             spatial= spatial2,
             iter = model$settings$iter, 
             step_size = model$settings$iter, 
             link = model$settings$link, 
             learning_rate = model$settings$learning_rate,
             device = model$settings$device
  )
  
  if(is.null(test)) {
    return(list(ll = logLik(m2), R=Rsquared(m2,...)))
  } else {
    
    m2$data$X = test_env 
    
    if(!is.null(spatial)) {
      
      m2$spatial$X = test_sp
      m2$data$Y = model$data$Y[test,,drop=FALSE]
      
      return(list(ll=m2$model$logLik(X=test_env,Y=model$data$Y[test,,drop=FALSE], SP=test_sp )[[1]], 
                  R=Rsquared(model=m2,...) ))
    } else {

      return(list(ll=m2$model$logLik(X=test_env,Y=model$data$Y[test,,drop=FALSE], SP=test_sp )[[1]], 
                  R=Rsquared(m2,X=test_env2, Y=model$data$Y[test,,drop=FALSE],...)))      
      
      
    }
  }
}


