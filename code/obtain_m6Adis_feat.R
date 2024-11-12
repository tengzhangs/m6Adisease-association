###m6A features
library(stringr)
f1 <- "F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\m6Asite_cosine_sim.csv"
m6A_cosine <- read.csv(f1)
colnames(m6A_cosine) <- c("source","target","sim_value")
m6A_cosine$source <- as.numeric(as.character(str_remove(m6A_cosine$source,"m6A")))
m6A_cosine$target <- as.numeric(as.character(str_remove(m6A_cosine$target,"m6A")))

m6A_feat <- data.frame()
for (i in 1:length(unique(m6A_cosine$source))) {
  options(warn = 1)
  onesite_feat <- m6A_cosine[m6A_cosine$source==i,]
  onesite_feat <- onesite_feat[order(onesite_feat$target,decreasing = F),]
  if(nrow(onesite_feat)==(length(unique(m6A_cosine$source))-1)){
    onesite_feat_value <- as.numeric(as.character(onesite_feat$sim_value))
  }  
  if(nrow(onesite_feat)<(length(unique(m6A_cosine$source))-1)){
    print(i)
    onesite_feat_value <- rep(0,(length(unique(m6A_cosine$source))-1))
    labels <- as.numeric(as.character(onesite_feat$target))
    new_label <- c(labels[which(labels<i)],(labels[which(labels>i)]-1))
    onesite_feat_value[new_label] <- as.numeric(as.character(onesite_feat$sim_value))
    
  }
  m6A_feat <- rbind(m6A_feat,onesite_feat_value)
}
colnames(m6A_feat) <- NULL
write.csv(m6A_feat,file = "F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\m6A_feature.csv",row.names = F)

new_m6Adis_sim <- read.csv(file = "F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\m6Adis_simi.csv")
colnames(new_m6Adis_sim) <- c("source","target","sim_value")
new_m6Adis_sim$source <- as.numeric(as.character(str_remove(new_m6Adis_sim$source,"dis")))
new_m6Adis_sim$target <- as.numeric(as.character(str_remove(new_m6Adis_sim$target,"dis")))

dis_feat <- data.frame()
for (i in 1:length(unique(new_m6Adis_sim$source))) {
  options(warn = 1)
  one_feat <- new_m6Adis_sim[new_m6Adis_sim$source==i,]
  if(nrow(one_feat)==(length(unique(new_m6Adis_sim$source))-1)){
    one_feat_value <- as.numeric(as.character(one_feat$sim_value))
  }  
  if(nrow(one_feat)<(length(unique(new_m6Adis_sim$source))-1)){
    one_feat_value <- rep(0,(length(unique(new_m6Adis_sim$source))-1))
    labels <- as.numeric(as.character(one_feat$target))
    new_label <- c(labels[which(labels<i)],(labels[which(labels>i)]-1))
    one_feat_value[new_label] <- as.numeric(as.character(one_feat$sim_value))
    print(i)
  }
  dis_feat <- rbind(dis_feat,one_feat_value)
}
colnames(dis_feat) <- NULL
write.csv(dis_feat,file = "F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\dis_feature.csv",row.names = F)
