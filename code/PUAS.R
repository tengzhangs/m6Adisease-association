library(xgboost)
library(stringr)
#####Transform the formation of m6A-disease associations
m6Adis_asso_infor <- function(known_asso,m6A_asso,dis_asso){
  m6Adis_asso_infor <- data.frame()
  for (i in 1:nrow(m6Adis_asso)) {
    
    pos_label <- which(m6Adis_asso[i,]==1)
    m6A_dis_noasso <- data.frame()
    for (j in 1:length(pos_label)) {
      one_dis_asso <- dis_asso[dis_asso$source==pos_label[j],]
      one_dis_asso <- one_dis_asso[order(one_dis_asso$target,decreasing = F),]
      if(length(which(!is.na(match(one_dis_asso$target,pos_label))))>0){
        one_dis_asso <- one_dis_asso[-which(!is.na(match(one_dis_asso$target,pos_label))),]
      }
      
      
      target_dis <- as.numeric(as.character(one_dis_asso$target))
      m6A_sites <- rep(i,length(target_dis))
      one_m6A_dis_noasso <- data.frame(m6A=m6A_sites,dis=target_dis,label=rep(0,length(target_dis)))
      
      m6A_dis_noasso <- rbind(m6A_dis_noasso,one_m6A_dis_noasso)
    }
    m6A_dis_noasso <- m6A_dis_noasso[order(m6A_dis_noasso$dis,decreasing = F),]
    m6A_dis_neg <- unique(m6A_dis_noasso$dis)
    neg_label <- rep(0,length(m6A_dis_neg))
    m6A_dis_infor <- c(pos_label,m6A_dis_neg)
    m6A_dis_infor <- m6A_dis_infor[order(m6A_dis_infor,decreasing = F)]
    pos_site <- which(!is.na(match(m6A_dis_infor,pos_label)))
    all_lable <- vector(length = length(m6A_dis_infor))
    all_lable[pos_site] <- 1
    one_asso_infor <- data.frame(m6A=rep(i,length(m6A_dis_infor)),dis=m6A_dis_infor,label=all_lable)
    m6Adis_asso_infor <- rbind(m6Adis_asso_infor,one_asso_infor)
  }
  return(m6Adis_asso_infor)
}
####Combine the embedding features for m6A sites and diseases

comb_feats <- function(m6Afeats,disfeats,m6Adis_asso_infor){
  colnames(m6Afeats) <- NULL
  colnames(disfeats) <- NULL
  new_m6A_dis_feat <- data.frame()
  for(i in 1:nrow(m6Afeats)){
    one_select_dis <- as.numeric(as.character(m6Adis_asso_infor[m6Adis_asso_infor$m6A==i,]$dis))
    one_m6A_dis_feat <- disfeats[one_select_dis,]
    one_m6A_feat <- m6Afeats[rep(i,nrow(one_m6A_dis_feat)),]
    rownames(one_m6A_feat) <- NULL
    onem6A_dis_feats <- cbind(one_m6A_feat,one_m6A_dis_feat)
    new_m6A_dis_feat <- rbind(new_m6A_dis_feat,onem6A_dis_feats)
  }
  new_label <- as.numeric(as.character(m6Adis_asso_infor$label))
  data=cbind(new_m6A_dis_feat,new_label)
  colnames(data) <- c(1:ncol(new_m6A_dis_feat),"Target_label")
  rownames(data) <- NULL
  return(data)
}

###Using XGboost model to train
xgboost_train <- function(Ps, Ns, dat, test=NULL, pos.probs=NULL, una.probs=NULL,  sampleFactor, seed,alpha){
  set.seed(seed)
  positive.train <- c()
  positive.cls <- c()
  # dat = data.mat
  # determine the proper sample size for creating a balanced dataset
  sampleN <- ifelse(length(Ps) < length(Ns), length(Ps), length(Ns))
  
  # bootstrap sampling to build the positive training set (labeled as 'P')
  idx.pl <- unique(sample(x=Ps, size=sampleFactor*sampleN, replace=T, prob=pos.probs[Ps]))
  # positive.train <- dat[Ps,]
  positive.train <- dat[idx.pl,]
  positive.cls <- rep("P", nrow(positive.train))
  
  # bootstrap sampling to build the "unannotate" or "negative" training set (labeled as 'N')
  idx.dl <- unique(sample(x=Ns, size=sampleFactor*sampleN, replace=T, prob=una.probs[Ns]))
  unannotate.train <- dat[idx.dl,]
  unannotate.cls <- rep("N", nrow(unannotate.train))
  
  # combine data
  train.sample <- rbind(positive.train, unannotate.train)
  rownames(train.sample) <- NULL;
  cls <- as.factor(c(positive.cls, unannotate.cls))
  
  ##xgboost
  xgb.fit = xgb.cv(data = train.sample, label =  ifelse(cls=="P",1,0), nrounds = 10 ,nfold=5,
                   objective = "binary:logistic", eta = 1,eval_metric ="auc",max_depth = 6,verbose = 0,early_stopping_rounds=3)
  elog <- as.data.frame(xgb.fit$evaluation_log)
  nround <- which.max(elog$test_auc_mean)
  
  xgb_fit = xgboost(data = train.sample, label =  ifelse(cls=="P",1,0), nrounds = nround ,
                    objective = "binary:logistic",max_depth = 6, eta = 1,eval_metric ="auc",verbose = 0)
  classification = predict(xgb_fit, dat)
  pro=classification
  pos_pro = pro[as.numeric(as.character(str_remove(Ps,"p_")))]
  cutoff = quantile(pos_pro, alpha)
  # rescale
  map.pred.y = pro - cutoff
  map.pred.y[map.pred.y>0] = map.pred.y[map.pred.y>0]/max(map.pred.y[map.pred.y>0])
  map.pred.y[map.pred.y<0] = map.pred.y[map.pred.y<0]/abs(min(map.pred.y[map.pred.y<0]))
  pred.y = 1/(1+exp(-10*(map.pred.y)))
  xgb_pred <- cbind(pred.y,1-pred.y)
  
  # xgb_pred <- cbind(pro,1-pro)
  colnames(xgb_pred) <- c("P","N")
  rownames(xgb_pred) <- rownames(dat)
  return(xgb_pred)
}

adpt_PU <- function(data,alpha,m6Adis_asso_infor){
  data.mat <- apply(X = data[,-ncol(data)], MARGIN = 2, FUN = as.numeric)
  data.cls <- new_label
  rownames(data.mat) <- paste("p", 1:nrow(data.mat), sep="_")
  set.seed(1)
  pos <- which(data.cls == 1)
  neg <- which(data.cls == 0)
  # Identify positive and negative examples 
  Ps <- rownames(data.mat)[which(data.cls == 1)]
  Ns <- rownames(data.mat)[which(data.cls == 0)]
  # train.mat=data.mat
  # test.mat=data.mat[which(!is.na(match(rownames(data.mat),Ns))),]
  
  
  # initialize sampling probablity
  pos.probs <- rep(1, length(Ps))
  una.probs <- rep(1, length(Ns))
  names(pos.probs) <- Ps
  names(una.probs) <- Ns
  
  sampleFactor = 1
  
  i <- 1
  
  
  pred_result <- list()
  
  xboost_result <- xgboost_train(Ps=Ps, Ns=Ns, dat=data.mat, pos.probs=pos.probs,
                                 una.probs=una.probs,  sampleFactor=sampleFactor,seed=1,alpha=alpha)
  pred_result[[1]] <- xboost_result
  while (i < 1000) {
    # update count
    i <- i + 1
    
    
    xboost_result <- xgboost_train(Ps=Ps, Ns=Ns, dat=data.mat, pos.probs=pos.probs,
                                   una.probs=una.probs,  sampleFactor=sampleFactor,seed=i,alpha=alpha)
    pos.probs <- xboost_result[Ps, "P"]
    una.probs <- xboost_result[Ns, "N"]
    pred_result[[i]] <- c(pos.probs,una.probs)
    
    delts <- mean(abs(pred_result[[i]]-pred_result[[i-1]]))
    print(delts)
    
    if(delts<0.01){
      break
    }
    
  }
  pos_pred <- pos.probs
  una_pred <- una.probs
  
  result <- vector(length = nrow(xboost_result))
  result[as.numeric(as.character(str_remove(Ps,"p_")))] <- 1
  select_Ns <- names(una_pred)[which(una_pred <0.5)]
  #select_PS <- names(pos_pred)[which(pos_pred>0.5)]
  result[as.numeric(as.character(str_remove(select_Ns,"p_")))] <- 1
  #result[as.numeric(as.character(str_remove(select_PS,"p_")))] <- 1
  last_m6Adis_asso <- data.frame(m6Adis_asso_infor,pred_label =result)
  return(last_m6Adis_asso)
}

PUAS_result <- function(raw_m6Adis_asso,m6A_asso,dis_asso,m6Afeats,disfeats){
  m6Adis_asso_infor <- m6Adis_asso_infor(known_asso=raw_m6Adis_asso,m6A_asso=m6A_asso,dis_asso=dis_asso)
  data <- comb_feats(m6Afeats=m6Afeats,disfeats=disfeats,m6Adis_asso_infor=m6Adis_asso_infor)
  PUAS_proc <- adpt_PU(data,alpha=0.1,m6Adis_asso_infor)
  return(PUAS_proc)
}

f1 <- "./m6Adis_asso.csv"
m6Adis_asso <- read.csv(f1)
m6Adis_asso <- m6Adis_asso[,-c(1:5)]
# label=as.vector(t(as.matrix(m6Adis_asso)))

f2 <- "./m6Asites_RWR.csv"
m6A_asso <- read.csv(f2)

f3 <- "./m6Adis_rwr.csv"
dis_asso <- read.csv(f3)
###
m6Afeats <- read.csv(file = "./m6A_GCN_embeding.csv")
colnames(m6Afeats) <- NULL
disfeats <- read.csv(file = "./disease_GCN_embeding.csv")
colnames(disfeats) <- NULL
PUAS_proc <- PUAS_result(raw_m6Adis_asso=m6Adis_asso,m6A_asso=m6A_asso,dis_asso=dis_asso,m6Afeats=m6Afeats,disfeats=disfeats)
write.csv(PUAS_proc,file = "./PUAS_process_m6Adis_asso.csv",row.names = F)
