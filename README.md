# Introduction
We propose a novel computational framework, m6A-Disease Prediction using Graph Convolutional Networks and Positive Unlabeled Learning with self-Adaptive Sampling (m6ADP-GCNPUAS), to accurately predict m6A-disease associations. m6ADP-GCNPUAS effectively captures the embedded features of 
m6A sites or diseases using GCN model. Given the very limited m6A-disease associations, m6ADP-GCNPUAS adopts the PUAS framework to augment the potential positive samples, which improves the predictive power for m6A-disease associations.
# Usage Example
## Step by Step 
### Calculate the similarity for m6A-m6A and disease-disease
#### Calculate the similarity for m6A sites

```r
#Get the sequences of m6A sites
fa <- "./m6Adis_asso.csv"
m6Adis_sites <- read.csv(fa)
library(GenomicRanges)

sites_GR <- GRanges(seqnames = as.character(m6Adis_sites$seqnames),
                  IRanges(start = as.numeric(as.character(m6Adis_sites$m6A_site)),
                          end = as.numeric(as.character(m6Adis_sites$m6A_site))),
                  strand = as.character(m6Adis_sites$strand))
sites_range <- resize(sites_GR, 501,fix="center")
library(BSgenome.Hsapiens.UCSC.hg19)
genome <- BSgenome.Hsapiens.UCSC.hg19
get_seq<- getSeq(genome,sites_range)
seqs <- as.character(get_seq)
write.table(seqs,file = "./m6Adis_seq.txt",row.names=F,col.names = F,quote=F)
```

```python
#Get the embedding representation of m6A sites and calculate the similarity between m6A sites
import logging
from gensim.models import  Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
import pandas as pd

seq_path="./m6Adis_seq.txt"
with open(seq_path,"r") as fr:
        lines = fr.readlines()
        
fr.close()
words=np.zeros(shape=(len(lines),499)).astype(np.str_)

i=0
k=3
for line in lines:
    j=0
    if line.startswith(">hsa") or len(line)<=1:
        continue
    else:
        line=line[:-1]
        seq_len=len(line)
        for index in range(0,seq_len,1):
            if index+k >= seq_len+1:
                break
            a=line[index:index+k]
            words[i,j]=a
            j=j+1
    i=i+1

pd.DataFrame(words).to_csv("./pos_m6a_dis_word_10.csv",index=False)

word_path="./pos_m6a_dis_word_10.txt"

with open(word_path,"w") as fw:
    for line in lines:
        if line.startswith(">hsa") or len(line)<=1:
                continue
        else:
            line=line[:-1]
            seq_len=len(line)
            for index in range(0,seq_len,1):
                if index+k>=seq_len+1:
                    break
                fw.write("".join(line[index:index+k]))
                fw.write(" ")
                fw.write("\n")
    fw.close()

#word2vec train

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",level=logging.INFO)

sentences=LineSentence("./pos_m6a_dis_word_10.txt")

    
vector_dim=100
model = Word2Vec(sentences, window=5, min_count=1, epochs=30, vector_size=vector_dim)
model.save("./nm6a_vec")
    
dataset = pd.read_csv("./pos_m6a_dis_word_10.csv")
word=model.wv.index_to_key
vector=model.wv.vectors

feature = np.zeros((len(lines),100,499))
    
for i in range(0,len(lines)):
    m=0        
    for j in range(0,499):
        char = dataset.iloc[i,j]
        index = word.index(char)
        feature[i,:,m] = vector[index,0:]
        m=m+1
        
features = np.zeros((len(lines),100))
features = np.sum(feature,axis=2)/499

pd.DataFrame(features).to_csv("./m6A_word2vec_feature.csv",index=False)
# calculate the simialrity of m6A sites by cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
m1=features
m1_similarity = cosine_similarity(m1)
pd.DataFrame(m1_similarity).to_csv("./m6A_cosine_similarity.csv",index=False)
```
#### Calculate the similarity for diseases-disease
```r
library(DOSE)
library(data.table)
f1 <- "./disease_IDinfor_new.xlsx"
DOID_infor <- readxl::read_xlsx(f1)
f2 <- "./m6Adis_asso.csv"
m6Adis_sites <- read.csv(f2)
match_m6Adis_name <- colnames(m6Adis_sites)[-c(1:5)]
newdis_infor <- data.frame()
for (i in 1:length(match_m6Adis_name)) {
  
  onedis_infor <- DOID_infor[DOID_infor$disease_name==match_m6Adis_name[i],]
  newdis_infor <- rbind(newdis_infor,onedis_infor)
}
a1 <- as.character(newdis_infor$ID)
a2 <- as.character(newdis_infor$ID)
s <- doSim(a1, a2, measure="Wang")
rownames(s) <- a1
s <- as.data.frame(s)
write.csv(s,file = "./dis_similariy.csv",row.names = F)
```
### Construct m6A-m6A interaction network and disease-disease interaction network by RWR

```r
m6A_sim <- read.csv(f1="./m6A_cosine_similarity.csv")
dis_sim <- read.csv(f2="./dis_similariy.csv"
m6A_m6A_net <- m6Asites_RWR(m6A_sim)
dis_dis_net <- dis_RWR(dis_sim)
```
### Extract the embedding feartures for m6A sites and diseases by GCN model

```python
#Runing the following python code
m6A_embeddingGCN.py
dis_embeddingGCN.py
```
### Augmenting positive samples by PUAS
```r
library(AdaSampling)
library(stringr)
library(xgboost)
#Input the raw known m6A-disease associtions
f1 <- "./m6Adis_asso.csv"
m6Adis_asso <- read.csv(f1)
m6Adis_asso <- m6Adis_asso[,-c(1:5)]
#Input the m6A-m6A interaction network and disease-disease interaction network
f2 <- "./m6A_m6A_net.csv"
f3 <- "./dis_dis_net.csv"
m6A_asso <- read.csv(f2)
dis_asso <- read.csv(f3)
#Input the embedding feature for m6A sites and diseases
m6Afeats <- read.csv(file = "./m6A_GCN_embeding.csv")
colnames(m6Afeats) <- NULL
disfeats <- read.csv(file = "./disease_GCN_embeding.csv")
colnames(disfeats) <- NULL
PUAS_procs <-  PUAS(raw_m6Adis_asso=m6Adis_asso,m6A_asso=m6A_asso,dis_asso=dis_asso,m6Afeats=m6Afeats,disfeats=disfeats)
write.csv(PUAS_proc,file = "./PUAS_process_m6Adis_asso.csv",row.names = F)
```
### Predicting m6A-disease associations by Random Forest
```r
library(randomForestSRC)
PUAS_result <-  read.csv(f1= "./PUAS_process_m6Adis_asso.csv")
new_label <- PUAS_result$pred_label
## Input features
m6Afeats <- read.csv(file = "./m6A_GCN_embeding.csv")
colnames(m6Afeats) <- NULL
disfeats <- read.csv(file = "./disease_GCN_embeding.csv")
colnames(disfeats) <- NULL
dataset <- data.frame(X,label=pred_label)
raw_label <- as.numeric(as.character(dataset$label))
dataset$label <- as.factor(ifelse(raw_label==1,"y","n"))
train_index <- sample(1:nrow(dataset), round(nrow(dataset) * 0.70))
  
train_pre <- rfsrc(label ~ .,data = dataset[train_index,],block.size = 1,nodesize = 1)
test_pre <- predict.rfsrc(train_pre, dataset[-train_index, ])
  
asso_pred_reslut <- test_pre[["predicted"]]
pred_value <- as.numeric(as.character(asso_pred_reslut[,2]))
test_label <- raw_label[-train_index]
pos_index <- which(test_label==1)
neg_index <- which(test_label==0)
fg <- asso_pred_reslut[pos_index,2]
bg <- asso_pred_reslut[neg_index,2]
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
# PR Curve
pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
conf_mat <- ModelMetrics::confusionMatrix(test_label,pred_value)
TN <- conf_mat[1,1]
TP <- conf_mat[2,2]
FP <- conf_mat[1,2]
FN <- conf_mat[2,1]
acc <- (TP + TN) / (TP + TN + FP + FN)
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
F1_score <- 2*(precision*recall)/(precision+recall)
```
