# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:31:16 2023

@author: tengz
"""



import logging
from gensim.models import  Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
import pandas as pd
'''
此代码块儿是分词过程
'''
seq_path="F:\\m6Adisease_data\\new_dis_m6A_data\\word2vect_cosine_sim\\m6Adis_seq.txt"
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

pd.DataFrame(words).to_csv("F:\\m6Adisease_data\\new_dis_m6A_data\\word2vect_cosine_sim\\pos_m6a_dis_word_10.csv",index=False)

word_path="F:\\m6Adisease_data\\new_dis_m6A_data\\word2vect_cosine_sim\\pos_m6a_dis_word_10.txt"

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

'''
word2vec train 
'''
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",level=logging.INFO)


sentences=LineSentence("F:\\m6Adisease_data\\new_dis_m6A_data\\word2vect_cosine_sim\\pos_m6a_dis_word_10.txt")

    
vector_dim=100
model = Word2Vec(sentences, window=5, min_count=1, epochs=30, vector_size=vector_dim)
model.save("F:\\m6Adisease_data\\new_dis_m6A_data\\word2vect_cosine_sim\\nm6a_vec")
    
dataset = pd.read_csv("F:\\m6Adisease_data\\new_dis_m6A_data\\word2vect_cosine_sim\\pos_m6a_dis_word_10.csv")
word=model.wv.index_to_key
vector=model.wv.vectors
    # feature=np.zeros((499,32))
#    features=[]
#    for idx, data in dataset.iterrows():
#        wv_feature = np.zeros((99, 100))
#        i=0
#        for ix,char in data.items():
#            wv_index=word.index(char)
#            wv_feature[i,:]=vector[wv_index]
#            i=i+1
#        features.append(wv_feature)
'''
根据每个位点的分词获得每个分词的embeding，然后将每个位点分词的embeding平均就和
'''
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

pd.DataFrame(features).to_csv("F:\\m6Adisease_data\\new_dis_m6A_data\\word2vect_cosine_sim\\m6A_word2vec_feature.csv",index=False)
# 计算features的m6A位点word2vec的cosine相似性
from sklearn.metrics.pairwise import cosine_similarity
m1=features
m1_similarity = cosine_similarity(m1)
        
pd.DataFrame(m1_similarity).to_csv("F:\\m6Adisease_data\\new_dis_m6A_data\\word2vect_cosine_sim\\m6A_cosine_similarity.csv",index=False)