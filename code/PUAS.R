library(xgboost)
library(stringr)
#f1 <- "F:\\m6Adisease_association_reasearch\\PU_learning_AdaSampling\\m6A_embeding.csv"
#f1 <-  "/home/disk3/zhangteng/m6A_disease/m6Asites_sim_infor/m6Asites_sim_PCA_infor.csv"
f1 <- "F:\\m6Adisease_data\\new_dis_m6A_data\\m6Asites_rwr.csv"
# m6Afeat <- read.csv(f1)
# colnames(m6Afeat) <- NULL
# #f2 <- "F:\\m6Adisease_association_reasearch\\PU_learning_AdaSampling\\dis_embeding.csv"
# f2 <- "F:\\m6Adisease_data\\new_dis_m6A_data\\m6Adis_rwr.csv"
# disfeat <- read.csv(f2)
# 
# 
# 
f3 <- "F:\\m6Adisease_data\\new_dis_m6A_data\\m6Adis_asso.csv"
m6Adis_asso <- read.csv(f3)
m6Adis_asso <- m6Adis_asso[,-c(1:5)]
label=as.vector(t(as.matrix(m6Adis_asso)))
# 
# nodes_feature <- function(num_node,node_feat){
#   colnames(node_feat) <- c("source" ,   "target" ,   "feature_value")
#   nodes_feat <- data.frame()
#   for (i in 1:num_node) {
#     one_site <- node_feat[node_feat$source==i,]
#     one_site <- one_site[order(one_site$target,decreasing = F),]
#     source_label <- seq(1,length(unique(node_feat$source)))
#     one_site_feat <- vector(length = length(source_label))
#     one_site_feat[as.numeric(as.character(one_site$target))] <- as.numeric(as.character(one_site$feature_value))
#     nodes_feat <- rbind(nodes_feat,one_site_feat)
#   }
#   colnames(nodes_feat) <- source_label
#   return(nodes_feat)
# }
# m6Afeats <- nodes_feature(num_node=nrow(m6Adis_asso),node_feat=m6Afeat)
# disfeats <- nodes_feature(num_node=ncol(m6Adis_asso),node_feat=disfeat)
##normalize

###
m6Afeats <- read.csv(file = "F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\m6A_feature.csv",header = F)
colnames(m6Afeats) <- NULL
disfeats <- read.csv(file = "F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\dis_feature.csv",header = F)
colnames(disfeats) <- NULL

RR=m6Afeats[rep(1:nrow(m6Afeats),each=nrow(disfeats)),]
dd=disfeats[rep(1:nrow(disfeats),nrow(m6Afeats)),]
feature=cbind(RR,dd)
data=cbind(feature,label)
colnames(data) <- c(1:ncol(feature),"Target_label")
rownames(data) <- NULL
library(AdaSampling)
data.mat <- apply(X = data[,-ncol(data)], MARGIN = 2, FUN = as.numeric)
data.cls <- label
rownames(data.mat) <- paste("p", 1:nrow(data.mat), sep="_")
set.seed(1)
pos <- which(data.cls == 1)
neg <- which(data.cls == 0)
# Identify positive and negative examples 
Ps <- rownames(data.mat)[which(data.cls == 1)]
Ns <- rownames(data.mat)[which(data.cls == 0)]
train.mat=data.mat
test.mat=data.mat[which(!is.na(match(rownames(data.mat),Ns))),]


# initialize sampling probablity
pos.probs <- rep(1, length(Ps))
una.probs <- rep(1, length(Ns))
names(pos.probs) <- Ps
names(una.probs) <- Ns

sampleFactor = 1

i <- 0
xgboost_train <- function(Ps, Ns, dat, test=NULL, pos.probs=NULL, una.probs=NULL,  sampleFactor, seed){
  set.seed(seed)
  positive.train <- c()
  positive.cls <- c()
  # dat = data.mat
  # determine the proper sample size for creating a balanced dataset
  sampleN <- ifelse(length(Ps) < length(Ns), length(Ps), length(Ns))
  
  # bootstrap sampling to build the positive training set (labeled as 'P')
  idx.pl <- unique(sample(x=Ps, size=sampleFactor*sampleN, replace=TRUE, prob=pos.probs[Ps]))
  # positive.train <- dat[Ps,]
  positive.train <- dat[idx.pl,]
  positive.cls <- rep("P", nrow(positive.train))
  
  # bootstrap sampling to build the "unannotate" or "negative" training set (labeled as 'N')
  idx.dl <- unique(sample(x=Ns, size=sampleFactor*sampleN, replace=TRUE, prob=una.probs[Ns]))
  unannotate.train <- dat[idx.dl,]
  unannotate.cls <- rep("N", nrow(unannotate.train))
  
  # combine data
  train.sample <- rbind(positive.train, unannotate.train)
  rownames(train.sample) <- NULL;
  cls <- as.factor(c(positive.cls, unannotate.cls))
  
  ##xgboost
  xgb.fit = xgboost(data = train.sample, label =  ifelse(cls=="P",1,0), nrounds = 3 ,
                    objective = "binary:logistic",max_depth = 10, eta = 1,eval_metric ="auc",verbose = 1)
  classification = predict(xgb.fit, dat)
  pro=classification
  pos_pro = pro[as.numeric(as.character(str_remove(Ps,"p_")))]
  cutoff = quantile(pos_pro, 0.1)
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

while (i < 10) {
  # update count
  i <- i + 1
  
  
  xboost_result <- xgboost_train(Ps=Ps, Ns=Ns, dat=data.mat, pos.probs=pos.probs,
                                 una.probs=una.probs,  sampleFactor=sampleFactor,seed=i)
  pos.probs <- xboost_result[Ps, "P"]
  una.probs <- xboost_result[Ns, "N"]
  ##KNN
  # knn.fit <- knn(train.sample, dat, cl=cls, k=5, prob=TRUE)
  # p <- attr(knn.fit, "prob")
  # idx <- which(knn.fit == "N")
  # p[idx] <- 1- p[idx]
  # knn.pred <- cbind(p, 1 - p)
  # colnames(knn.pred) <- c("P", "N")
  # rownames(knn.pred) <- rownames(dat)
  # length(which(knn.pred[,1]>0.5))/nrow(knn.pred)
  # update probability arrays
  # pos.probs <- knn.pred[Ps, "P"]
  # una.probs <- knn.pred[Ns, "N"]
}
pred <- xgboost_train(Ps=Ps, Ns=Ns, dat=data.mat, pos.probs=pos.probs,
                      una.probs=una.probs,  sampleFactor=sampleFactor,seed=1)
pos_pred <- pred[Ps,'P']
una_pred <- pred[Ns,"N"]
length(which(pos_pred>0.5))
length(which(una_pred >0.5))
result <- vector(length = nrow(pred))
result[as.numeric(as.character(str_remove(Ps,"p_")))] <- 1
select_Ns <- names(una_pred)[which(una_pred >0.5)]
result[as.numeric(as.character(str_remove(select_Ns,"p_")))] <- 1
probability=t(matrix(result,ncol=1367,nrow=457))
Adsampling_XGboostresult <- list(pred_result=pred,new_label =probability)
save(Adsampling_XGboostresult,file = "/home/disk3/zhangteng/m6A_disease/PU_learning/Adsampling_XGboostresult.Rdata")
