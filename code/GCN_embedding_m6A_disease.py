# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:11:24 2024

@author: tengz
"""
import dgl
import dgl.nn  as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import itertools
from random import sample 
import pandas as pd
import numpy as np
import scipy.sparse as sp
import dgl.function as fn


class GNN_MLP_model(nn.Module):
    #定义GNN层以及相关参数
    def __init__(self, 
                 in_feats, hid_feats_1, hid_feats_2, 
                 mlp_hid_feats_1, mlp_out_feats,
                 dropout):
        super().__init__()
        #GNN更新节点特征
        self.SAGE1 = dglnn.SAGEConv(in_feats=in_feats,
                                    out_feats=hid_feats_1,
                                    aggregator_type="mean")
        self.SAGE2 = dglnn.SAGEConv(in_feats=hid_feats_1,
                                    out_feats=hid_feats_2,
                                    aggregator_type="mean")
        #MLP更新边的特征
        self.MLP1 = nn.Linear(hid_feats_2*2, mlp_hid_feats_1)
        self.MLP2 = nn.Linear(mlp_hid_feats_1, mlp_out_feats)
        
        self.dropout = nn.Dropout(dropout)
    
    def apply_edges(self, edges):
        h_u,  h_v = edges.src['h'], edges.dst['h']
        h_concat = torch.cat([h_u, h_v], 1)
        h2 = F.relu(self.MLP1(h_concat))
        h2 = self.dropout(h2)
        #与边回归的主要区别之一：本质上是二分类问题
        h2 = torch.sigmoid(self.MLP2(h2))
        return {'score':h2}
         
    #定义前向传播函数:需要输入图结构、节点特征
    def forward(self, graph, inputs, edge_graph):
        # GNN部分
        h = F.relu(self.SAGE1(graph, inputs))
        h = self.dropout(h)
        h = F.relu(self.SAGE2(graph, h))
        # MLP部分
        with edge_graph.local_scope():
            edge_graph.ndata['h'] = h
            edge_graph.apply_edges(self.apply_edges)
            return edge_graph.edata['score'], h

import scipy.sparse as sp
import numpy as np
def produce_5_graph(graph, train_mask, test_mask):
    u, v = graph.edges()
    eids = np.arange(graph.number_of_edges())
    
    #用于更新节点的图(排除测试集的边)
    train_g = dgl.remove_edges(graph, eids[test_mask])
    
    #阳性边的训练集/测试集
    train_pos_g = dgl.graph((u[eids[train_mask]], v[eids[train_mask]]), 
                        num_nodes=graph.number_of_nodes())
    test_pos_g = dgl.graph((u[eids[test_mask]], v[eids[test_mask]]), 
                            num_nodes=graph.number_of_nodes())
    
    #阴性边的训练集/测试集
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(graph.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    
    neg_eids = np.random.choice(len(neg_u), graph.number_of_edges(), replace=False)
    train_neg_g = dgl.graph((neg_u[neg_eids[train_mask]], neg_v[neg_eids[train_mask]]), 
                            num_nodes=graph.number_of_nodes())
    test_neg_g = dgl.graph((neg_u[neg_eids[test_mask]], neg_v[neg_eids[test_mask]]), 
                            num_nodes=graph.number_of_nodes())
    
    return train_g, train_pos_g, test_pos_g, train_neg_g, test_neg_g

from sklearn.metrics import roc_auc_score
def loss_evaluate(model, train_g, pos_g, neg_g, mode="train"):
    if mode=="train":
        model.train()
    else:
        model.eval()
    pos_score,pos_embeding = model(train_g, train_g.ndata[feature_name], pos_g)
    neg_score,neg_embeding = model(train_g, train_g.ndata[feature_name], neg_g)
    scores = torch.cat([pos_score, neg_score]).reshape((-1))
    labels = torch.cat([torch.ones(pos_score.shape[0]),  
                        torch.zeros(neg_score.shape[0])]) 
    loss = F.binary_cross_entropy(scores, labels)
    auc = roc_auc_score(labels.detach().numpy(), scores.detach().numpy())
    return loss, auc, pos_embeding

def train(model, graph, 
          feature_name, train_mask, test_mask,
          num_epochs, learning_rate, weight_decay, patience, verbose=True):
    train_g, train_pos_g, test_pos_g, train_neg_g, test_neg_g = produce_5_graph(graph, train_mask, test_mask)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    val_loss_best = 100000
    trigger_times = -1
    
    for epoch in range(num_epochs):
        loss, auc, train_pos_embeding = loss_evaluate(model, train_g, train_pos_g, train_neg_g)
        test_loss,test_auc,test_pos_embeding =  loss_evaluate(model, train_g, test_pos_g, test_neg_g)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose :
            print("Epoch {:03d} | Loss {:.4f} | Auc {:.4f} | Test Loss {:.4f} | Test Auc {:.4f} ".format(
                                  epoch, loss.item(), auc, test_loss.item(), test_auc))
        
        if test_loss.item() > val_loss_best:
            trigger_times += 1
            if trigger_times >= patience:
                break
        else:
            trigger_times = 0
            val_loss_best = test_loss.item()
       
    return loss.item(), auc, test_loss.item(), test_auc,train_pos_embeding


dis_asso = pd.read_csv("F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\m6Adis_rwr.csv")
                                                                         
dis_asso['source']=dis_asso['source']-1
dis_asso['target']=dis_asso['target']-1

# m6Adis_asso = pd.read_csv("F:\\m6Adisease_data\\new\\m6Adis_asso_infor.csv")
m6A_asso = pd.read_csv("F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\m6Asites_RWR.csv")
m6A_asso['source']=m6A_asso['source']-1
m6A_asso['target']=m6A_asso['target']-1

m6A_feat = pd.read_csv("F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\m6A_feature.csv",header=None,dtype=np.float32)
m6Afeat_tensor = torch.from_numpy(m6A_feat.to_numpy())
#m6A_feat_L2 = m6A_feat.apply(np.linalg.norm, axis=1)
#m6A_feat_norm=m6A_feat.div(m6A_feat_L2,axis=0)

dis_feat = pd.read_csv("F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\dis_feature.csv",header=None,dtype=np.float32)
dis_feat_tensor = torch.from_numpy(dis_feat.to_numpy())
#dis_feat_L2 = dis_feat.apply(np.linalg.norm,axis=1)
#dis_feat_norm = dis_feat.div(dis_feat_L2,axis=0)

#m6A_rows, m6A_cols = dis_feat.shape[0], dis_feat.shape[1]
#dis_rows, dis_cols = dis_feat.shape[0], dis_feat.shape[1]




m6A_edges_src = torch.from_numpy(m6A_asso['source'].to_numpy())
m6A_edges_dst = torch.from_numpy(m6A_asso['target'].to_numpy())

dis_edges_src = torch.from_numpy(dis_asso['source'].to_numpy())
dis_edges_dst = torch.from_numpy(dis_asso['target'].to_numpy())


asso_u = m6A_edges_src
asso_v = m6A_edges_dst
'''
adj = sp.coo_matrix((np.ones(len(asso_u)), (asso_u.numpy(), asso_v.numpy())))
adj_A = adj.A
adj_neg = 1 - adj.todense()
neg_u, neg_v = np.where(adj_neg != 0)
neg_u = torch.from_numpy(neg_u)
neg_v = torch.from_numpy(neg_v)
all_u = torch.cat((asso_u,neg_u),dim=0)
all_v = torch.cat((asso_v,neg_v),dim=0)
labels = torch.cat((torch.ones(len(asso_u)),torch.zeros(len(neg_u))),dim=0)
'''
m6A_g = dgl.graph((asso_u, asso_v), num_nodes=m6Afeat_tensor.shape[0])
m6A_g.ndata['feature'] = m6Afeat_tensor
n_features = m6Afeat_tensor.shape[1]
# m6A_g.edata['label'] =labels
dis_g = dgl.graph((dis_edges_src,dis_edges_dst),num_nodes=dis_feat_tensor.shape[0])
dis_g.ndata['feature'] = dis_feat_tensor
n_features = dis_feat_tensor.shape[1]


feature_name = "feature"

graph = m6A_g

graph=dis_g


# 按同一种边划分出训练集与测试集
train_mask = torch.zeros(graph.num_edges(),dtype=torch.bool).bernoulli(0.7)
test_mask =  ~ train_mask


in_feats, hid_feats_1, hid_feats_2, mlp_hid_feats_1, mlp_out_feats, dropout = n_features, 200, 100, 100, 1, 0
num_epochs, learning_rate, weight_decay, patience = 100, 0.001, 1e-4, 5

#实例化模型
model = GNN_MLP_model(in_feats, hid_feats_1, hid_feats_2, mlp_hid_feats_1, mlp_out_feats, dropout)
#训练模型
metrics = train(model, graph, 
                feature_name, train_mask, test_mask,
                num_epochs, learning_rate, weight_decay, patience, verbose=True)

train_pos_embeding = metrics[4]
embeding_array = train_pos_embeding.detach().numpy() 
df = pd.DataFrame(embeding_array) 

df.to_csv('F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\disease_GCN_embeding.csv',  index=False) 

df.to_csv('F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\m6A_GCN_embeding.csv',  index=False) 



###生成第i/k折的训练集与测试集掩码
eids = np.arange(graph.number_of_edges())
def get_k_fold_data(k, j, edge_id, shuffle=True):
    assert k > 1
    train_mask = torch.ones(len(eids), dtype=torch.bool)
    np.random.seed(42)
    if shuffle:
        edge_id2 = np.random.permutation(range(len(train_mask)))
    else :
        edge_id2 = train_mask
        
    fold_size = len(train_mask) // k
    idx = slice(j*fold_size, (j+1)*fold_size)
    train_mask[edge_id2[idx]] = False
    
    #train_mask = train_mask.tile((2,))
    test_mask = ~ train_mask
    return train_mask, test_mask

def k_fold(k, model, graph,
           feature_name, train_mask, test_mask,
           num_epochs, learning_rate, weight_decay, patience):
    k_fold_metrics = []  #收集每一折的结果
    edge_id = np.arange(graph.number_of_edges())
    for j in range(k):
        print(f'Fold-{j+1}')
        train_mask, test_mask = get_k_fold_data(k, j, edge_id)
        model = GNN_MLP_model(in_feats, hid_feats_1, hid_feats_2, mlp_hid_feats_1, mlp_out_feats, dropout)
        metrics = train(model, graph, 
                        feature_name, train_mask, test_mask,
                        num_epochs, learning_rate, weight_decay, patience, verbose=True)
        
        k_fold_metrics.append(metrics)
    return k_fold_metrics

k = 5
in_feats, hid_feats_1, hid_feats_2, mlp_hid_feats_1, mlp_out_feats, dropout = n_features, 200, 100, 100, 1, 0.01
num_epochs, learning_rate, weight_decay, patience = 100, 0.001, 1e-4, 5

k_fold_metrics = k_fold(k, model, graph, 
                        feature_name, train_mask, test_mask,
                        num_epochs, learning_rate, weight_decay, patience)
np.array(k_fold_metrics).mean(0)