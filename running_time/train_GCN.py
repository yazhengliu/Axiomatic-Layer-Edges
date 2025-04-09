import argparse
import numpy as np
import torch
import random
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Coauthor,Planetoid
import sys
import copy
from torch_geometric.nn import MessagePassing
import math
import torch.nn as nn
from torch_geometric.utils import add_self_loops,dense_to_sparse
import scipy.sparse as sp
import torch.optim as optim
from torch.optim import SGD
from torch import Tensor
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.nn import Parameter
import cvxpy as cvx
from torch.nn.utils import parameters_to_vector

import time

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid, normalize=False, add_self_loops=False, bias=False)
        self.gc2 = GCNConv(nhid, nclass, normalize=False, add_self_loops=False, bias=False)
        # self.gc3 = GraphConvolution(nhid2, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # adj=torch.tensor(adj,requires_grad=True)
        # adj = F.relu(adj)

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.gc3(x, adj)
        # print(F.softmax(x, dim=1))
        return x

    def back(self, x, adj):
        x0 = self.gc1(x, adj)
        x1 = F.relu(x0)
        x2 = self.gc2(x1, adj)
        return (x0, x1, x2)

def matrixtodict(nonzero): # 将邻接矩阵变为字典形式存储

    a = []
    graph = dict()
    for i in range(0, len(nonzero[1])):
        if i != len(nonzero[1]) - 1:
            if nonzero[0][i] == nonzero[0][i + 1]:
                a.append(nonzero[1][i])
            if nonzero[0][i] != nonzero[0][i + 1]:
                a.append(nonzero[1][i])
                graph[nonzero[0][i]] = a
                a = []
        if i == len(nonzero[1]) - 1:
            a.append(nonzero[1][i])
        graph[nonzero[0][len(nonzero[1]) - 1]] = a
    return graph
def normalize(mx): #卷积算子
    rowsum = np.array(mx.sum(1)) #行求和
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0. #将稀疏矩阵之中每行全为0的替换
    r_mat_inv = sp.diags(r_inv) #产生对角阵
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)  #卷积算子
    return mx

def split_train_test():
    node_labels = dict()
    for i in range(0, labels.max().item() + 1):
        node_labels[i] = []
    labelslist = list(range(labels.max().item() + 1))

    # labelslist=[0,1,2,3,4,5,6,7]
    for i in range(0, len(labels)):
        for j in range(0, len(labelslist)):
            if labels[i].item() == labelslist[j]:
                node_labels[labelslist[j]] = node_labels[labelslist[j]] + [i]
    # print(node_labels)
    idx_train = []
    idx_val = []
    idx_test = []
    for i in range(0, len(labelslist)):
        random.shuffle(node_labels[i])
        train_slice = node_labels[i][:math.floor(train_ratio * len(node_labels[i]))]
        val_slice = node_labels[i][math.floor(train_ratio * len(node_labels[i])):math.floor(
            (train_ratio + val_ratio) * len(node_labels[i]))]
        test_slice = node_labels[i][math.floor((train_ratio + val_ratio) * len(node_labels[i])):]
        idx_train = idx_train + train_slice
        idx_val = idx_val + val_slice
        idx_test = idx_test + test_slice
    return idx_train,idx_val,idx_test


def construct_adj_matrix(edge_index,labels):
    edges = []
    for idx ,node in enumerate(edge_index[0]):
        edges.append((node,edge_index[1][idx]))
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj)
    return adj
def forward_tensor(adj, layernumbers, W):  # 有relu
    hiddenmatrix = dict()
        # adj = torch.tensor(adj, requires_grad=True)
        # adj=sparse_mx_to_torch_sparse_tensor(adj)
    relu = torch.nn.ReLU(inplace=False)
    hiddenmatrix[0] = W[0]

    h = torch.sparse.mm(adj, W[0])

    hiddenmatrix[1] = torch.mm(h, W[1])
    hiddenmatrix[2] = relu(hiddenmatrix[1])
        # hiddenmatrix[1].retain_grad()
    for i in range(1, layernumbers):
        h = torch.sparse.mm(adj, hiddenmatrix[2 * i])
        hiddenmatrix[2 * i + 1] = torch.mm(h, W[i + 1])
        if i != layernumbers - 1:
            hiddenmatrix[2 * i + 2] = relu(hiddenmatrix[2 * i + 1])
            # hiddenmatrix[i + 1].retain_grad()
    return hiddenmatrix
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  #转换为coo矩阵
    indices = torch.from_numpy( np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
#
def train(epoch):
    t = time.time()
    model.train()
    # 清空前面的导数缓存
    optimizer.zero_grad()
    output = model(x, adj_sp)
    # print('----------------------------')
    # print(output)
    # print('----------------------------')
    # print(labels)
    # entroy=torch.nn.CrossEntropyLoss()
    # loss_train = entroy(output[idx_train], labels[idx_train])
    # acc_train = accuracy(output[idx_train], labels[idx_train])
    # loss_train = F.cross_entropy(output[data.train_mask], labels[data.train_mask])
    # acc_train = accuracy(output[data.train_mask], labels[data.train_mask])

    loss_train = F.cross_entropy(output[torch.tensor(idx_train)], labels[torch.tensor(idx_train)])
    # loss_fair_train=losszong_mean(group_dict, output, 'train')

    total_loss=loss_train
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[torch.tensor(idx_train)], labels[torch.tensor(idx_train)])
    #求导
    total_loss.backward()
    #更新
    optimizer.step()

    model.eval()  #不启用dropout
    output = model(x, adj_sp)


    # loss_val = entroy(output[idx_val], labels[idx_val])
    loss_val=F.cross_entropy(output[torch.tensor(idx_val)],labels[torch.tensor(idx_val)])
    # # loss_val = F.nll_loss(output[idx_val], labels[idx_val])


    acc_val = accuracy(output[torch.tensor(idx_val)], labels[torch.tensor(idx_val)])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),

          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),

          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden1', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--hidden2', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1-keep probability)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)



    # dataset = Coauthor('../data/Physics', 'Physics')
    # modelname = 'physics'

    dataset = Coauthor('../data/Cs', 'Cs')
    modelname = 'cs'

    # dataset = Planetoid('../data/pubmed', 'PubMed')
    # modelname = 'pubmed'


    data=dataset[0]
    print(data)
    x, edge_index = data.x, data.edge_index
    labels = data.y
    model = GCN(nfeat=x.shape[1],
                nhid=args.hidden1,
                nclass=labels.max().item() + 1,
                dropout=args.dropout
                )
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)


    # model.eval()
    # model.load_state_dict(torch.load(f'../{modelname}GCNmodel.pkl'))

    train_ratio = 0.5
    val_ratio = 0.3
    idx_train, idx_val, idx_test = split_train_test()

    edges_old = copy.deepcopy(edge_index.tolist())
    # for i in range(x.shape[0]):
    #     edges_old[0].append(i)
    #     edges_old[1].append(i)



    adj_sp = construct_adj_matrix(edges_old, labels)
    print('adj_sp', adj_sp)

    # adj_sp = adj_sp + sp.eye(len(labels))
    print('adj_sp',adj_sp[0,0])

    if (adj_sp.todense() == adj_sp.todense().T).all():
        print("adj_start是对称矩阵。")
    else:
        print("adj_start不是对称矩阵。")



    adj_old_nonzero = adj_sp.nonzero()
    graph = matrixtodict(adj_old_nonzero)
    adj = adj_sp.todense()
    adj_sp = sparse_mx_to_torch_sparse_tensor(adj_sp)



    numclass = labels.max().item() + 1
    group_dict = dict()

    for epoch in range(args.epochs):
        train(epoch)

    torch.save(model.state_dict(), f'../data/{modelname}/{modelname}_GCNmodel.pkl')
    model.eval()
    model.load_state_dict(torch.load(f'../data/{modelname}/{modelname}_GCNmodel.pkl'))
    output = model(x, adj_sp)

    loss1 = F.cross_entropy(output[torch.tensor(idx_train)], labels[torch.tensor(idx_train)])

    acc_val = accuracy(output[torch.tensor(idx_val)], labels[torch.tensor(idx_val)])
    test_acc=accuracy(output[torch.tensor(idx_test)], labels[torch.tensor(idx_test)])


    print('acc_val',acc_val,'test_acc',test_acc)
