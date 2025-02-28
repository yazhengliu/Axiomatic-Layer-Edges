from torch_geometric.data import Data,InMemoryDataset, DataLoader
import torch
import os.path as osp
import pandas as pd
import networkx as nx
import numpy as np
import cvxpy as cvx
import os
import copy
import time,datetime
import scipy.sparse as sp
import argparse
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import math
from torch import Tensor
class Net_link(torch.nn.Module):
    def __init__(self,nfeat,nhid):
        super(Net_link, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid,add_self_loops=False,normalize=False,bias=False)
        self.conv2 = GCNConv(nhid, nhid,add_self_loops=False,normalize=False,bias=False)
        self.linear = nn.Linear(nhid * 2, 2,bias=False)
        # self.MLP1 = nn.Linear(args.hidden * 2, args.mlp_hidden)
        # self.MLP2 = nn.Linear(args.mlp_hidden, 2)

    def encode(self, x,edge_index1,edge_index2,edge_weight1,edge_weight2):
        x = self.conv1(x.to(torch.float32), edge_index1,edge_weight=edge_weight1)
        x = x.relu()
        return self.conv2(x, edge_index2,edge_weight=edge_weight2)



    def decode(self, z, pos_edge_index):
        edge_index = pos_edge_index

        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        h=self.linear(h)

        return h

    def forward(self,x, edge_index1,edge_index2,edge_weight1,edge_weight2,pos_edge_index):
        z = self.encode(x, edge_index1,edge_index2,edge_weight1,edge_weight2)
        z=self.decode(z,pos_edge_index)
        return z

    def back_MLP(self, z, pos_edge_index):
        edge_index = pos_edge_index

        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        h = self.linear(h)
        return h

    def back(self, x, edge_index_1, edge_index_2, edgeweight1, edgeweight2):
        x_0 = self.conv1(x, edge_index_1, edge_weight=edgeweight1)
        x_1 = F.relu(x_0)
        return (x_0, x_1)

def split_edge(start,end,flag,clear_time,num_nodes):
    edge_index = [[], []]
    max_month=0
    max_week=0
    print('start',start)
    print('end',end)
    if flag == 'year':
        for key, value in clear_time.items():
            if value[0] >= start and value[0] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])
    if flag == 'month':
        for key, value in clear_time.items():
            max_month=max(max_month,value[1])
            if value[1] >= start and value[1] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])

    if flag=='week':
        for key, value in clear_time.items():
            max_month = max(max_month, value[1])
            max_week = max(max_week, value[2])
            if value[2] >= start and value[2] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])
    # print('max_month',max_month)
    # print('max_week', max_week)

    # for i in range(num_nodes):
    #     edge_index[0].append(i)
    #     edge_index[1].append(i)
    return edge_index
def clear_time(time_dict):
    edge_time = dict()
    max_month=0

    for key, value in time_dict.items():
        month = (value.year - 2010) * 12 + value.month
        max_month=max(month,max_month)
        week = (value.year - 2010) * 52 + value.isocalendar()[1]
        edge_time[key]=(value.year,month,week)
    clear_time = dict()
    for key, value in edge_time.items():
        if (key[1], key[0]) in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
        if (key[1], key[0]) not in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
    print('max_month',max_month)
    return clear_time
def clear_time_UCI(time_dict):
    edge_time = dict()
    max_week = 0

    for key, value in time_dict.items():
        # print('value.year',value.year)
        month = (value.year - 2004) * 12 + value.month
        week = (value.year - 2004) * 52 + value.isocalendar()[1]
        max_week = max(week, max_week)
        edge_time[key]=(value.year,month,week)
    clear_time = dict()
    for key, value in edge_time.items():
        if (key[1], key[0]) in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
        if (key[1], key[0]) not in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
    # print('max_week',max_week)
    return clear_time

def link_load_data(path):
    # data_dir = 'data'
    # data_csv = 'bitcoinotc.csv'
    # filename = os.path.join(data_dir, data_csv)
    df = pd.read_csv(path)
    # print(df)
    Graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(df, source='SOURCE', target='TARGET', edge_attr='RATING', create_using=Graphtype)

    mapping = {}
    count = 0
    for node in list(G.nodes):
        mapping[node] = count
        count = count + 1
    G = nx.relabel_nodes(G, mapping)

    rating = nx.get_edge_attributes(G, 'RATING')
    # print('rating',rating)
    max_rating = rating[max(rating, key=rating.get)]
    degree_sequence_in = [d for n, d in G.in_degree()]
    dmax_in = max(degree_sequence_in)
    degree_sequence_out = [d for n, d in G.out_degree()]
    dmax_out = max(degree_sequence_out)
    # print(A)
    # if (6002,6000) in G.edges():
    #     print('yes')
    # else:
    #     print('false')

    # print(len(G.edges()))
    # print(len(edges_index[0]))

    feat_dict = {}
    feature_length = 8
    for node in list(G.nodes):
        out_edges_list = G.out_edges(node)
        # print('out_edges_list',out_edges_list)

        if len(out_edges_list) == 0:
            features = np.ones(feature_length, dtype=float) / 1000
            feat_dict[node] = {'feat': features}
        else:
            features = np.zeros(feature_length, dtype=float)
            w_pos = 0
            w_neg = 0
            for (_, target) in out_edges_list:
                w = G.get_edge_data(node, target)['RATING']
                if w >= 0:
                    w_pos = w_pos + w
                else:
                    w_neg = w_neg - w

            abstotal = (w_pos + w_neg)
            average = (w_pos - w_neg) / len(out_edges_list) / max_rating

            features[0] = w_pos / max_rating / len(out_edges_list)  # average positive vote
            features[1] = w_neg / max_rating / len(out_edges_list)  # average negative vote
            features[2] = w_pos / abstotal
            features[3] = average
            features[4] = features[0] * G.in_degree(node) / dmax_in
            features[5] = features[1] * G.in_degree(node) / dmax_in
            features[6] = features[0] * G.out_degree(node) / dmax_out
            features[7] = features[1] * G.out_degree(node) / dmax_out

            features = features / 1.01 + 0.001

            feat_dict[node] = {'feat': features}
    nx.set_node_attributes(G, feat_dict)
    G = G.to_undirected()
    # print(G.edges())
    A = nx.adjacency_matrix(G).todense()
    X = np.asarray([G.nodes[node]['feat'] for node in list(G.nodes)])
    edges_index = [[], []]
    for edge in G.edges():
        edges_index[0].append(edge[0])
        edges_index[1].append(edge[1])
        edges_index[1].append(edge[0])
        edges_index[0].append(edge[1])
    # for i in range(max(edges_index[1])+1):
    #     edges_index[0].append(i)
    #     edges_index[1].append(i)

    time_dict = dict()
    df = df.values
    for i in range(0, df.shape[0]):
        edge_0 = df[i][0]
        edge_1 = df[i][1]
        t1 = datetime.datetime.utcfromtimestamp(df[i][3])
        time_dict[(mapping[edge_0], mapping[edge_1])] = t1
    # print(A.shape)
    # print(X)
    # print(X.shape)
    return edges_index,X,mapping,time_dict #边 节点特征
def link_read_data(folder: str, prefix):
    # path=os.path.join(folder, f"{prefix}.npz")
    # data_csv = 'bitcoinotc.csv'
    path = os.path.join(folder, f"{prefix}.csv")
    print(path)
    edges_index,X ,mapping,time_dict= link_load_data(path)

    # x = torch.from_numpy(features).float()
    # y = torch.from_numpy(labels)
    # print('y',y)
    features=torch.DoubleTensor(X)

    edge_index = torch.LongTensor(edges_index)
    print('ed',edge_index)
    data = Data(x=features,  edge_index=edge_index,node_map=mapping,time_dict=time_dict)
    # node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    # node_mask= torch.zeros(adj.shape[0], dtype=torch.bool)
    # train_mask=node_mask.clone()
    # val_mask = node_mask.clone()
    # test_mask = node_mask.clone()
    # print('node_mask',node_mask)
    # for i in range(0,adj.shape[0]):
    #     if i in idx_train:
    #         # print('train')
    #         train_mask[i]=True
    #     if i in idx_test:
    #         # print('test')
    #         test_mask[i]=True
    #     if i in idx_val:
    #         # print('val')
    #         val_mask[i]=True
    #
    # data.train_mask = train_mask
    # data.val_mask = val_mask
    # data.test_mask = test_mask
    return data
class SynGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(SynGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"{self.name}.csv"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data = link_read_data(self.root, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
class Net_link_evaulate(torch.nn.Module):
    def __init__(self,nfeat,nhid):
        super(Net_link_evaulate, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid,add_self_loops=False,normalize=False,bias=False)
        self.conv2 = GCNConv(nhid, nhid,add_self_loops=False,normalize=False,bias=False)
        self.linear = nn.Linear(nhid * 2, 2,bias=False)
        # self.MLP1 = nn.Linear(args.hidden * 2, args.mlp_hidden)
        # self.MLP2 = nn.Linear(args.mlp_hidden, 2)

    def encode(self, x,edge_index,edge_weight):
        # print(type(data.x))
        #
        # print(data.x.type())

        x = self.conv1(x.to(torch.float32), edge_index,edge_weight=edge_weight)
        x = x.relu()
        return self.conv2(x, edge_index,edge_weight=edge_weight)



    def decode(self, z, pos_edge_index):
        edge_index = pos_edge_index
        # print('edge_index',edge_index)
        # print('len',len(pos_edge_index[0])+len(neg_edge_index[1]))
        # print(max(edge_index[0]))
        # print(max(edge_index[1]))
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        h=self.linear(h)
        # h = self.MLP1(h)
        # h = h.relu()
        # h = self.MLP2(h)

        # h=h.sum(dim=-1)
        # print('h', h.shape)
        # logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        # print('logits.shape',logits.shape)
        return h

    def forward(self,x, edge_index,edge_weight,pos_edge_index):
        z = self.encode(x, edge_index,edge_weight)
        z=self.decode(z,pos_edge_index)
        return z

    def back_MLP(self, z, pos_edge_index):
        edge_index = pos_edge_index
        # print('edge_index',edge_index)
        # print('len',len(pos_edge_index[0])+len(neg_edge_index[1]))
        # print(max(edge_index[0]))
        # print(max(edge_index[1]))
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        h = self.linear(h)
        return h

    def back(self, x, edge_index_1, edge_index_2, edgeweight1, edgeweight2):
        x_0 = self.conv1(x, edge_index_1, edge_weight=edgeweight1)
        x_1 = F.relu(x_0)
        return (x_0, x_1)

