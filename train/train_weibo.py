import pandas as pd
import numpy as np
from dateutil.parser import parse
from datetime import datetime
import time
import json
import math
import os
import random
import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import argparse
import torch.optim as optim
import re
import torch.nn as nn
from nltk.corpus import stopwords
import nltk.stem
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import os, re, csv, math, codecs,string
from torch_geometric.nn import GCNConv


class Net_rumor(torch.nn.Module):
    def __init__(self, nhid, nclass,dropout,args):
        super(Net_rumor, self).__init__()
        self.conv1 = GCNConv(nhid*2, nhid,add_self_loops=False,bias=False,normalize=False)
        # self.conv2 = GCNConv(nhid1, dataset.num_classes)
        self.conv2=GCNConv(nhid, nclass,add_self_loops=False,bias=False,normalize=False)
        # self.conv3 = GCNConv(nhid2, dataset.num_classes)
        # self.conv4 = GCNConv(nhid3, dataset.num_classes)
        self.dropout = dropout
        # self.linear = nn.Linear(768, nfeat)
        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(args.glove_embedding, requires_grad=False)
        self.bilstm = nn.LSTM(input_size=embed_dim, hidden_size=args.hidden,
                              batch_first=True, num_layers=args.num_layers, bidirectional=True)  # bidirectional=True



    def forward(self, sentence, edge_index_1, edge_index_2,edgeweight1,edgeweight2):
        # print(self.conv1(x, edge_index))
        x = self.embed(sentence)
        # print(x)
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        _, (hidden, cell) = self.bilstm(x)
        # x=x[-1, :, :]
        x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # x=self.linear(x)
        x = F.relu(self.conv1(x, edge_index_1,edge_weight=edgeweight1))
        # print('x[goal]',x[goal])

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index_2,edge_weight=edgeweight2)

        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.conv3(x,edge_index)
        return x
        # return F.log_softmax(x, dim=1)

    def back(self, x, edge_index_1, edge_index_2,edgeweight1,edgeweight2):
        x_0 = self.conv1(x, edge_index_1,edge_weight=edgeweight1)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index_2,edge_weight=edgeweight2)
        return (x_0, x_1)

    def feature(self, sentence):
        x = self.embed(sentence)
        # print(x)
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        _, (hidden, cell) = self.bilstm(x)
        # x=x[-1, :, :]
        x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return x

    def forward_v2(self, x, edge_index_1, edge_index_2,edgeweight1,edgeweight2):
        # print(self.conv1(x, edge_index))

        # x=self.linear(x)
        x = F.relu(self.conv1(x, edge_index_1,edge_weight=edgeweight1))
        # print('x[goal]',x[goal])

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index_2,edge_weight=edgeweight2)

        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.conv3(x,edge_index)
        return x
# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
#
#         self.weight = Parameter(torch.Tensor(in_channels, out_channels))
#         self.register_parameter('bias', None)
#         self.reset_parameters()
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#
#     def forward(self, x, edge_index):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]
#
#         # Step 1: 增加自连接到邻接矩阵
#         # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#
#         # Step 2: 对节点的特征矩阵进行线性变换
#         x = x @ self.weight
#
#         # Step 3-5: Start propagating messages.
#         return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
#
#     def message(self, x_j, edge_index, size):
#         # x_j has shape [E, out_channels]
#
#         # Step 3: Normalize node features.
#         # row, col = edge_index
#         # deg = degree(row, size[0], dtype=x_j.dtype)
#         # deg_inv_sqrt = deg.pow(-0.5)
#         # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#
#         return  x_j
#
#     def update(self, aggr_out):
#         # aggr_out has shape [N, out_channels]
#
#         # Step 5: Return new node embeddings.
#         return aggr_out
class Net(torch.nn.Module):
    def __init__(self, nhid, nclass,dropout,args):
        super(Net, self).__init__()
        self.conv1 = GCNConv(nhid*2, nhid,normalize=True,add_self_loops=False,bias=False)
        # self.conv2 = GCNConv(nhid1, dataset.num_classes)

        self.conv2=GCNConv(nhid, nclass,normalize=True,add_self_loops=False,bias=False)
        # self.conv3 = GCNConv(nhid2, dataset.num_classes)
        # self.conv4 = GCNConv(nhid3, dataset.num_classes)
        self.dropout = dropout
        # self.linear = nn.Linear(768, nfeat)
        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(args.glove_embedding, requires_grad=False)
        self.bilstm = nn.LSTM(input_size=embed_dim, hidden_size=args.hidden,
                              batch_first=True, num_layers=args.num_layers, bidirectional=True)  # bidirectional=True



    def forward(self, sentence, edge_index_1, edge_index_2):
        # print(self.conv1(x, edge_index))
        x = self.embed(sentence)
        # print(x)
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        _, (hidden, cell) = self.bilstm(x)
        # x=x[-1, :, :]
        x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # x=self.linear(x)
        x = F.relu(self.conv1(x, edge_index_1))
        # print('x[goal]',x[goal])

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index_2)

        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.conv3(x,edge_index)
        return x
        # return F.log_softmax(x, dim=1)

    def back(self, x, edge_index_1, edge_index_2):
        x_0 = self.conv1(x, edge_index_1)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index_2)
        return (x_0, x_1)

    def feature(self, sentence):
        x = self.embed(sentence)
        # print(x)
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        _, (hidden, cell) = self.bilstm(x)
        # x=x[-1, :, :]
        x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return x

    def forward_v2(self, x, edge_index_1, edge_index_2):
        # print(self.conv1(x, edge_index))

        # x=self.linear(x)
        x = F.relu(self.conv1(x, edge_index_1))
        # print('x[goal]',x[goal])

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index_2)

        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.conv3(x,edge_index)
        return x
# def accuracy(preds, labels):
#     # preds = output.max(1)[1].type_as(labels)
#     correct = preds.eq(labels).double()
#     correct = correct.sum()
#     return correct / len(labels)
def accuracy_list(preds, labels):
    # preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def train_all(args):
    jsonPath = f'data/weibo/weibo_word_index.json'

    with open(jsonPath, 'r') as f:
        word_index = json.load(f)
    print('word_index success')
    embedding_numpy = np.load("data/weibo/weibo_embedding.npy")
    print('embedding_numpy', embedding_numpy)
    print(' embedding_numpy success')
    embedding_tensor = torch.FloatTensor(embedding_numpy)

    args.glove_embedding = embedding_tensor


    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    pheme_clean_path = 'data/weibo/weibo_json/'
    files_name = [file.split('.')[0] for file in os.listdir(pheme_clean_path)]
    file_map = dict()
    for i in range(0, len(files_name)):
        file_map[files_name[i]] = i
    # print(file_map)
    idx_list = list(range(max(file_map.values()) + 1))
    file_map_reverse = {value: key for key, value in file_map.items()}
    label_list = []
    for i in range(0, len(idx_list)):
        if i % 1000 == 0:
            print('i', i)
        file_name = file_map_reverse[i]
        jsonPath = f'data/weibo/weibo_json/{file_name}.json'
        with open(jsonPath, 'r') as f:
            data = json.load(f)
        label_list.append(data['label'])

    label_list_tensor = torch.LongTensor(label_list)

    idx_label_0 = []
    idx_label_1 = []
    for i in range(0, len(label_list)):
        if label_list[i] == 0:
            idx_label_0.append(i)
        if label_list[i] == 1:
            idx_label_1.append(i)
    random.shuffle(idx_label_0)
    random.shuffle(idx_label_1)

    # print(idx_list)
    train_ratio = 0.6
    val_ratio = 0.2

    # random.shuffle(idx_list)
    train_list_0 = idx_label_0[0:math.floor(len(idx_label_0) * train_ratio)]

    val_list_0 = idx_label_0[
                 math.floor(len(idx_label_0) * train_ratio):math.floor(len(idx_label_0) * (train_ratio + val_ratio))]
    test_list_0 = idx_label_0[math.floor(len(idx_label_0) * (train_ratio + val_ratio)):]

    train_list_1 = idx_label_1[0:math.floor(len(idx_label_1) * train_ratio)]

    val_list_1 = idx_label_1[
                 math.floor(len(idx_label_1) * train_ratio):math.floor(len(idx_label_1) * (train_ratio + val_ratio))]
    test_list_1 = idx_label_1[math.floor(len(idx_label_1) * (train_ratio + val_ratio)):]
    train_list = train_list_0 + train_list_1
    val_list = val_list_0 + val_list_1
    test_list = test_list_0 + test_list_1
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)

    #
    # label_train = []
    # label_val = []
    # label_test = []
    #
    # # print(train_list)
    # # print(test_list)
    # print(val_list)
    print(len(train_list))
    print(len(val_list))
    print(len(test_list))
    #
    #
    #
    model = Net(
        nhid=args.hidden,
        nclass=2,
        dropout=args.dropout, args=args)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    batch_size = 32
    batch_list = []
    for i in range(0, len(train_list)):
        if i % batch_size == 0:
            batch_list.append(i)
    if len(train_list) not in batch_list:
        batch_list.append(len(train_list))
    print(batch_list)
    print(len(train_list))

    batch_list=[batch_list[0],batch_list[1]]
    data_prefix = 'weibo'

    best_acc = 0
    early_stop_step = 10
    temp_early_stop_step = 0
    data_prefix = 'weibo'

    # for epoch in range(args.epochs):
    #     train(epoch,model,batch_list,file_map_reverse,train_list,label_list_tensor,optimizer)
    #     # val()
    #     if epoch % 1 == 0:
    #         temp_acc = test(model,test_list,file_map_reverse)
    #         if temp_acc > best_acc:
    #             print('save model')
    #             best_acc = temp_acc
    #             torch.save(model.state_dict(), f'data/{data_prefix}/' + data_prefix + '_GCN_model.pth')
    #         temp_early_stop_step = 0
    #     else:
    #         temp_early_stop_step += 1
    #         if temp_early_stop_step >= early_stop_step:
    #             print('early stop')
    #             break
    # # torch.save(model.state_dict(), f'data/{data_prefix}/'+data_prefix + '_GCN_model.pth')

    model_path=f'data/{data_prefix}/{data_prefix}_GCN_model.pth'
    model.load_state_dict(torch.load(model_path))
    val(model,val_list,file_map_reverse)

    test(model,test_list,file_map_reverse)

def train(epoch,model,batch_list,file_map_reverse,train_list,label_list_tensor,optimizer):
    t = time.time()
    model.train()
    # optimizer.zero_grad()

    label_train=[]
    label_train_pred=[]
    avg_loss=[]

    for batch_index in range(0,len(batch_list)-1):
        print('batch',batch_list[batch_index])
        loss_train = 0
        for train_index in range(batch_list[batch_index],batch_list[batch_index+1]):
            file_name = file_map_reverse[train_list[train_index]]
            jsonPath = f'data/weibo/weibo_json/{file_name}.json'

            with open(jsonPath, 'r') as f:
                data = json.load(f)
            sentence = np.array(data['intput sentenxe'])
            sentence = torch.LongTensor(sentence)
            # print('sentence',sentence)
            # print('sentence', sentence.shape)

            edges_index = data['edges_3']
            edges_index_tensor = torch.tensor(edges_index)
            output = model(sentence, edges_index_tensor, edges_index_tensor)
            label_train.append(data['label'])
            label_train_pred.append(torch.unsqueeze(output[0], 0).max(1)[1].item())
            loss = F.cross_entropy(torch.unsqueeze(output[0], 0), label_list_tensor[train_list[train_index]].view(-1))
            loss_train = loss + loss_train
            avg_loss.append(loss.item())


        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    avg_loss = np.array(avg_loss).mean()
    label_train = torch.tensor(label_train)
    label_train_pred = torch.tensor(label_train_pred)
    # print('label_true',label_train)
    # print('label_pred',label_train_pred)
    acc_train = accuracy_list(label_train_pred, label_train)
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss: {:.4f}'.format(avg_loss),
          'acc_train: {:.4f}'.format(acc_train.item()),
          )
    return avg_loss,acc_train


def val(model,val_list,file_map_reverse):
    loss_val = 0
    label_val = []
    label_val_pred = []
    model.eval()
    avg_loss=[]
    for val_idx in val_list:
        val_index = val_list.index(val_idx)
        if val_index % 100 == 0:
            print(val_index)
        file_name = file_map_reverse[val_idx]
        jsonPath = f'data/weibo/weibo_json/{file_name}.json'

        with open(jsonPath, 'r') as f:
            data = json.load(f)
        sentence = np.array(data['intput sentenxe'])
        sentence = torch.LongTensor(sentence)

        edges_index = data['edges_3']
        edges_index_tensor = torch.tensor(edges_index)

        output = model(sentence, edges_index_tensor, edges_index_tensor)
        # loss = F.cross_entropy(torch.unsqueeze(output[0], 0), label_list_tensor[val_idx].view(-1))
        # # print(loss)
        # loss_val = loss + loss_val
        label_val_pred.append(torch.unsqueeze(output[0], 0).max(1)[1].item())
        label_val.append(data['label'])
        # avg_loss.append(loss.item())

    label_val = torch.tensor(label_val)
    label_val_pred = torch.tensor(label_val_pred)
    acc_val = accuracy_list(label_val_pred, label_val)
    avg_loss = np.array(avg_loss).mean()
    print(
          'loss_val: {:.4f}'.format(avg_loss),
          'acc_val: {:.4f}'.format(acc_val.item()),
          )

def test(model,test_list,file_map_reverse):
    model.eval()
    loss_test=0
    label_test = []
    label_test_pred = []
    avg_loss=[]
    for test_idx in test_list:
        test_index = test_list.index(test_idx)
        if test_index % 100 == 0:
            print(test_index)
        file_name = file_map_reverse[test_idx]
        jsonPath = f'data/weibo/weibo_json/{file_name}.json'

        with open(jsonPath, 'r') as f:
            data = json.load(f)
        sentence = np.array(data['intput sentenxe'])
        sentence = torch.LongTensor(sentence)

        edges_index = data['edges_3']
        edges_index_tensor = torch.tensor(edges_index)
        output = model(sentence, edges_index_tensor, edges_index_tensor)
        # loss = F.cross_entropy(torch.unsqueeze(output[0], 0), label_list_tensor[test_idx].view(-1))
        # # print(loss)
        # loss_test = loss + loss_test
        # # print(torch.unsqueeze(output[0], 0).max(1)[1].item())
        label_test_pred.append(torch.unsqueeze(output[0], 0).max(1)[1].item())

        # print(edges_index)
        label_test.append(data['label'])
        # avg_loss.append(loss.item())


    label_test = torch.tensor(label_test)
    label_test_pred = torch.tensor(label_test_pred)
    acc_test=accuracy_list(label_test_pred, label_test)
    avg_loss = np.array(avg_loss).mean()
    print(
          'loss_test: {:.4f}'.format(avg_loss),
          'acc_test: {:.4f}'.format(acc_test.item()),
         )
    return acc_test.item()



