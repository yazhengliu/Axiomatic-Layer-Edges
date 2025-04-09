import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
import random
import json
import os
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
import math
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCN
from tqdm import tqdm
import torch
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from torch_geometric.utils import degree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph

from torch_geometric.nn import GCN
from torch_geometric.utils import k_hop_subgraph
import torch
from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self, nfeat,hidden_channels,nclass):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        # self.weight=Parameter(torch.Tensor(nfeat, hidden_channels),requires_grad=True)
        self.conv1 = GCNConv(nfeat, hidden_channels,add_self_loops=False,normalize=False,bias=False)
        self.conv2 = GCNConv(hidden_channels, nclass,add_self_loops=False,normalize=False,bias=False)
        # self.embedding=nn.Embedding(40, hidden_channels)
        # self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, edge_weight):
        # 1. 获得节点嵌入
        # x=[i for i in range(x.shape[0])]
        #
        # x=self.embedding(torch.tensor(x))

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)

        # 2. Readout layer
        # print(batch)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        # print('batch',batch)
        # print('x',x.mean(dim=0, keepdim=True) )
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # print('x',x.shape)

        # 3. 分类器
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin(x)
        return x

    def back(self, x, edge_index, edge_weight):
        x_0 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index, edge_weight=edge_weight)
        return x_0, x_1
    def pre_forward(self,x, edge_index,edge_weight):



        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)

        return x

    def verify_layeredge(self, x, edge_index1, edge_index2, edge_weight1, edge_weight2):
        # print(self.conv1(x, edge_index))
        x = F.relu(self.conv1(x, edge_index1, edge_weight=edge_weight1))

        # x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv2(x, edge_index2, edge_weight=edge_weight2)

        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        # print('batch',batch)
        # print('x',x.mean(dim=0, keepdim=True) )
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        return x
def train():
    model.train()
    optimizer.zero_grad()
    loss=0
    for i in range(train_number):
        # print('idx',i)
        json_path = data_path + '/' + str(i) + '/' + 'positive.json'
        with open(json_path, 'r') as f:
            result = json.load(f)

        x=np.array(result['x'])
        x=torch.tensor(x)
        edge_index=result['edge_index']
        edge_index=torch.tensor(edge_index)

        edge_weight=result['edge_weight']
        edge_weight=torch.tensor(edge_weight)

        # print('x',x)


        # print('data.edge_index',data.edge_index)
        # print('data.x', data.x)
        x=x.to(torch.float32)
        # print(x.shape[1])
        out = model(x, edge_index, edge_weight)
        # print('out',out)
        # print('label',data.y)
        loss += criterion(out, torch.tensor([1]))

        json_path = data_path + '/' + str(i) + '/' + 'negative.json'
        with open(json_path, 'r') as f:
            result = json.load(f)

        x = np.array(result['x'])
        x = torch.tensor(x)
        edge_index = result['edge_index']
        edge_index = torch.tensor(edge_index)
        edge_weight = result['edge_weight']
        edge_weight = torch.tensor(edge_weight)

        # print('data.edge_index',data.edge_index)
        # print('data.x', data.x)
        x = x.to(torch.float32)
        out = model(x, edge_index, edge_weight)
        loss += criterion(out, torch.tensor([0]))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    loss.backward()
    print('loss',loss)
    optimizer.step()
def acc_train():
    model.eval()

    correct = 0

    for i in range(train_number):
        json_path = data_path + '/' + str(i) + '/' + 'positive.json'
        with open(json_path, 'r') as f:
            result = json.load(f)

        x = np.array(result['x'])
        x = torch.tensor(x)
        edge_index = result['edge_index']
        edge_index = torch.tensor(edge_index)
        edge_weight = result['edge_weight']
        edge_weight = torch.tensor(edge_weight)

        # print('data.edge_index',data.edge_index)
        # print('data.x', data.x)
        x = x.to(torch.float32)
        out = model(x, edge_index, edge_weight)

        pred = out.argmax(dim=1)  # 使用概率最高的类别

        correct += int((pred == torch.tensor([1])).sum())
        # print('train 1 pred', pred)

        json_path = data_path + '/' + str(i) + '/' + 'negative.json'
        with open(json_path, 'r') as f:
            result = json.load(f)

        x = np.array(result['x'])
        x = torch.tensor(x)
        edge_index = result['edge_index']
        edge_index = torch.tensor(edge_index)
        edge_weight = result['edge_weight']
        edge_weight = torch.tensor(edge_weight)

        # print('data.edge_index',data.edge_index)
        # print('data.x', data.x)
        x = x.to(torch.float32)
        out = model(x, edge_index, edge_weight)
        pred = out.argmax(dim=1)  # 使用概率最高的类别
        # print('train 0 pred',pred)

        correct += int((pred == torch.tensor([0])).sum())


    # print('correct',correct)
    return correct/train_number/2
def acc_val():
    model.eval()

    correct = 0

    for i in range(train_number,val_number):
        json_path = data_path + '/' + str(i) + '/' + 'positive.json'
        with open(json_path, 'r') as f:
            result = json.load(f)

        x = np.array(result['x'])
        x = torch.tensor(x)
        edge_index = result['edge_index']
        edge_index = torch.tensor(edge_index)
        edge_weight = result['edge_weight']
        edge_weight = torch.tensor(edge_weight)

        # print('data.edge_index',data.edge_index)
        # print('data.x', data.x)
        x = x.to(torch.float32)
        out = model(x, edge_index, edge_weight)

        pred = out.argmax(dim=1)  # 使用概率最高的类别

        correct += int((pred == torch.tensor([1])).sum())

        # print('1,pred',pred)

        json_path = data_path + '/' + str(i) + '/' + 'negative.json'
        with open(json_path, 'r') as f:
            result = json.load(f)

        x = np.array(result['x'])
        x = torch.tensor(x)
        edge_index = result['edge_index']
        edge_index = torch.tensor(edge_index)
        edge_weight = result['edge_weight']
        edge_weight = torch.tensor(edge_weight)

        # print('data.edge_index',data.edge_index)
        # print('data.x', data.x)
        x = x.to(torch.float32)
        out = model(x, edge_index, edge_weight)
        pred = out.argmax(dim=1)  # 使用概率最高的类别
        # print('0 pred',pred)

        correct += int((pred == torch.tensor([0])).sum())

    # print('correct',correct)
    return correct/(val_number-train_number)/2


if __name__=='__main__':

    changed_ratio=0.9
    data_type='circle'
    data_path=f'gen_{data_type}_data/{changed_ratio}'
    data_path_list=os.listdir(data_path)
    print(data_path_list)
    train_number=500
    val_number = 1000

    model = GCN(nfeat=11, hidden_channels=20, nclass=2)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.005)
    criterion = torch.nn.CrossEntropyLoss()


    for epoch in range(1, 2000):
        train()
        train_acc = acc_train()
        test_acc = acc_val()
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    torch.save(model.state_dict(), f'{data_type}_GCN_model_{changed_ratio}.pth')
    model.eval()

    model_path = f'{data_type}_GCN_model_{changed_ratio}.pth'
    model.load_state_dict(torch.load(model_path))
    test_acc = acc_val()
    print(test_acc)


    # for a in range(train_number):
    #     json_path = data_path+ '/' + a+'/'+'positive.json'
    #     with open(json_path, 'r') as f:
    #         result = json.load(f)
    #     print(result)


    # idx = torch.arange(data.num_nodes)
    # train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=data.y)
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data = data.to(device)
    # model = GCN(data.num_node_features, hidden_channels=20, num_layers=3,
    #             out_channels=dataset.num_classes).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)


    # def train():
    #     model.train()
    #     optimizer.zero_grad()
    #     out = model(data.x, data.edge_index)
    #     loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    #     loss.backward()
    #     optimizer.step()
    #     return float(loss)
    #
    #
    # @torch.no_grad()
    # def test():
    #     model.eval()
    #     pred = model(data.x, data.edge_index).argmax(dim=-1)
    #
    #     train_correct = int((pred[train_idx] == data.y[train_idx]).sum())
    #     train_acc = train_correct / train_idx.size(0)
    #
    #     test_correct = int((pred[test_idx] == data.y[test_idx]).sum())
    #     test_acc = test_correct / test_idx.size(0)
    #
    #     return train_acc, test_acc
    #
    #
    # pbar = tqdm(range(1, 2001))
    # for epoch in pbar:
    #     loss = train()
    #     if epoch == 1 or epoch % 200 == 0:
    #         train_acc, test_acc = test()
    #         pbar.set_description(f'Loss: {loss:.4f}, Train: {train_acc:.4f}, '
    #                              f'Test: {test_acc:.4f}')
    # pbar.close()
    # model.eval()