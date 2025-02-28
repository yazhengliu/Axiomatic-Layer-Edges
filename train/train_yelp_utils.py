import pickle
import numpy as np
import torch
import  scipy.sparse as sp
from sklearn import metrics
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch import Tensor
import cvxpy as cvx
import copy
def read_data(tvt, urp, city_name, train_ratio, val_ratio):
    #    tvt = 'train' / 'val' / 'test'
    #    urp = 'user' / 'review' / 'prod'
    test_ratio = str(round(100 * (1 - train_ratio - val_ratio)))
    train_ratio = str(int(100 * train_ratio))
    val_ratio = str(int(100 * val_ratio))
    with open(f'data/{city_name}/' + tvt + '_' + urp + '_' + city_name + '_' + train_ratio + val_ratio + test_ratio, 'rb') as f:
        #          str(int(100*train_ratio)) +
        #          str(int(100*val_ratio)) +
        #          str(round(100*test_ratio)), 'rb') as f:
        nodelist = pickle.load(f)
    return nodelist
def read_user_prod(review_list):
    user_list =[]
    prod_list =[]
    # for x in review_list:
    #     if x[0] not in user_list:
    #         user_list.append(x[0])
    #     if x[1] not in prod_list:
    #         prod_list.append(x[1])
    user_list = list(set([x[0] for x in review_list]))
    prod_list = list(set([x[1] for x in review_list]))
    user_list=sorted(user_list)
    prod_list=sorted(prod_list)
    return user_list, prod_list
def seperate_r_u(features, idx_list, l_idx, l_fea, l_nums, temp):
    r_idx = []
    r_fea = []
    u_idx = []
    u_fea = []
    for idx in idx_list:
        # print('idx',idx)
        if isinstance(idx, tuple):
            r_idx.append(idx)
            r_fea.append(features[idx])
        elif idx[0] == 'u':
            u_idx.append(idx)
            u_fea.append(features[idx])
    l_idx += (r_idx + u_idx)
    l_fea += (r_fea + u_fea)
    l_nums.append([len(r_idx) + temp,
                   len(r_idx) + len(u_idx) + temp])
    temp += len(r_idx) + len(u_idx)
    return l_idx, l_fea, l_nums, temp
def feature_matrix(features, p_train, p_val, p_test):
    l_idx = []
    l_fea = []
    l_nums = []

    temp = 0
    l_idx, l_fea, l_nums, temp = seperate_r_u(features, p_train, l_idx, l_fea, l_nums, temp)
    l_idx, l_fea, l_nums, temp = seperate_r_u(features, p_val, l_idx, l_fea, l_nums, temp)
    l_idx, l_fea, l_nums, temp = seperate_r_u(features, p_test, l_idx, l_fea, l_nums, temp)

    # features_list=list(features.keys())
    # print(features_list)
    # prod_idx=[]
    # for idx in features_list:
    #     if idx not in p_train and idx not in p_val and idx not in p_test:
    #         prod_idx.append(idx)
    prod_idx = list(set(list(features.keys())) - set(p_train) - set(p_val) - set(p_test))
    prod_idx=sorted(prod_idx)
    # print('prod_idx',prod_idx)
    prod_fea = []
    for idx in prod_idx:
        prod_fea.append(features[idx])

    l_idx += prod_idx
    l_fea += prod_fea
    l_nums.append([len(p_train),
                   len(p_train) + len(p_val),
                   len(p_train) + len(p_val) + len(p_test),
                   len(p_train) + len(p_val) + len(p_test) + len(prod_idx)])
    return l_idx, l_fea, l_nums
def onehot_label(ground_truth, list_idx):
    labels = np.zeros((len(list_idx), 2))

    gt = {}
    user_gt = {}
    for k, v in ground_truth.items():
        u = k[0]
        p = k[1]
        if u not in gt.keys():
            gt[u] = v
            user_gt[u] = v
        else:
            gt[u] |= v #update
            user_gt[u] |= v
        if p not in gt.keys():
            gt[p] = v
        else:
            gt[p] |= v
    ground_truth = {**ground_truth, **gt}

    for it, k in enumerate(list_idx):
        labels[it][ground_truth[k]] = 1
    return labels, user_gt

def construct_adj_matrix(ground_truth, idx_map, labels,rev_time,time1,time2,flag):
    edges = []
    # print(ground_truth.keys())
    keys_list=list(ground_truth.keys())
    if flag=='month':
        for it, r_id in enumerate(ground_truth.keys()):
            if rev_time[r_id][1] >= time1 and rev_time[r_id][1]<time2 :
                edges.append((idx_map[r_id], idx_map[r_id[0]]))
                edges.append((idx_map[r_id], idx_map[r_id[1]]))

                edges.append((idx_map[r_id[0]], idx_map[r_id]))
                edges.append((idx_map[r_id[1]], idx_map[r_id]))
    if flag == 'week':
        for it, r_id in enumerate(ground_truth.keys()):
            # print('r_id',r_id)
            if rev_time[r_id][2] < time2 and rev_time[r_id][2] >= time1:
                edges.append((idx_map[r_id], idx_map[r_id[0]]))
                edges.append((idx_map[r_id], idx_map[r_id[1]]))

                edges.append((idx_map[r_id[0]], idx_map[r_id]))
                edges.append((idx_map[r_id[1]], idx_map[r_id]))
        # for i in range(0, len(keys_list)):
        #     r_id = keys_list[i]
        #     if  rev_time[r_id][2] >= time1 and rev_time[r_id][2]<time2 :
        #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
        #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
        #
        #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
        #         edges.append((idx_map[r_id[1]], idx_map[r_id]))
    if flag == 'year':
        # for i in range(0, len(keys_list)):
        #     r_id = keys_list[i]
        #     if rev_time[r_id][0] < time2 and rev_time[r_id][0] >= time1:
        #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
        #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
        #
        #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
        #         edges.append((idx_map[r_id[1]], idx_map[r_id]))
        for it, r_id in enumerate(ground_truth.keys()):
            # print('r_id',r_id)
            if rev_time[r_id][0]< time2 and rev_time[r_id][0]>=time1:
                edges.append((idx_map[r_id], idx_map[r_id[0]]))
                edges.append((idx_map[r_id], idx_map[r_id[1]]))

                edges.append((idx_map[r_id[0]], idx_map[r_id]))
                edges.append((idx_map[r_id[1]], idx_map[r_id]))
    # for i in range(0,len(keys_list)):
    #     r_id=keys_list[i]
    #     if rev_time[r_id]< year2 and rev_time[r_id]>=year1:
    #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
    #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
    #
    #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
    #         edges.append((idx_map[r_id[1]], idx_map[r_id]))




    edges = np.array(edges)
    # print('edges',edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj=adj + sp.eye(adj.shape[0])
    adj=normalize(adj)
    # print(adj[105650,105650])
    #print('adj', adj)
    return adj
def normalize(mx): #卷积算子
    rowsum = np.array(mx.sum(1)) #行求和
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0. #将稀疏矩阵之中每行全为0的替换
    r_mat_inv = sp.diags(r_inv) #产生对角阵
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)  #卷积算子
    return mx
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def auc_score(output, ground_truth, list_idx, idx_range, u_or_r):
    prob = torch.exp(output[:, 1]).detach().numpy()
    prob_dic = {}
    for it, idx in enumerate(list_idx):
        prob_dic[idx] = prob[it]
    sub_list = [list_idx[x] for x in idx_range]
    sub_true = []
    sub_prob = []
    if u_or_r == 'r':
        for x in sub_list:
            if isinstance(x, tuple):
                sub_true.append(ground_truth[x])
                sub_prob.append(prob_dic[x])
    elif u_or_r == 'u':
        for x in sub_list:
            if isinstance(x, str) and x[0] == 'u':
                sub_true.append(ground_truth[x])
                sub_prob.append(prob_dic[x])
    fpr, tpr, thre = metrics.roc_curve(sub_true, sub_prob)



    return metrics.auc(fpr, tpr)
def construct_edge(ground_truth, idx_map, labels, rev_time, time1, time2, flag):
    edges = [[], []]

    # print(ground_truth.keys())
    keys_list = list(ground_truth.keys())
    if flag == 'month':
        for it, r_id in enumerate(ground_truth.keys()):
            if rev_time[r_id][1] >= time1 and rev_time[r_id][1] < time2:
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[0]])
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[1]])
                edges[0].append(idx_map[r_id[1]])
                edges[1].append(idx_map[r_id])
                edges[0].append(idx_map[r_id[0]])
                edges[1].append(idx_map[r_id])
                # edges.append((idx_map[r_id], idx_map[r_id[0]]))
                # edges.append((idx_map[r_id], idx_map[r_id[1]]))
                #
                # edges.append((idx_map[r_id[0]], idx_map[r_id]))
                # edges.append((idx_map[r_id[1]], idx_map[r_id]))
    if flag == 'week':
        for it, r_id in enumerate(ground_truth.keys()):
            # print('r_id',r_id)
            if rev_time[r_id][2] < time2 and rev_time[r_id][2] >= time1:
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[0]])
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[1]])
                edges[0].append(idx_map[r_id[1]])
                edges[1].append(idx_map[r_id])
                edges[0].append(idx_map[r_id[0]])
                edges[1].append(idx_map[r_id])
        # for i in range(0, len(keys_list)):
        #     r_id = keys_list[i]
        #     if  rev_time[r_id][2] >= time1 and rev_time[r_id][2]<time2 :
        #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
        #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
        #
        #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
        #         edges.append((idx_map[r_id[1]], idx_map[r_id]))
    if flag == 'year':
        # for i in range(0, len(keys_list)):
        #     r_id = keys_list[i]
        #     if rev_time[r_id][0] < time2 and rev_time[r_id][0] >= time1:
        #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
        #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
        #
        #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
        #         edges.append((idx_map[r_id[1]], idx_map[r_id]))
        for it, r_id in enumerate(ground_truth.keys()):
            # print('r_id',r_id)
            if rev_time[r_id][0] < time2 and rev_time[r_id][0] >= time1:
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[0]])
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[1]])
                edges[0].append(idx_map[r_id[1]])
                edges[1].append(idx_map[r_id])
                edges[0].append(idx_map[r_id[0]])
                edges[1].append(idx_map[r_id])
    # for i in range(0,len(keys_list)):
    #     r_id=keys_list[i]
    #     if rev_time[r_id]< year2 and rev_time[r_id]>=year1:
    #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
    #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
    #
    #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
    #         edges.append((idx_map[r_id[1]], idx_map[r_id]))

    # edgeitself = list(range(labels.shape[0]))
    # for it, edge in enumerate(edgeitself):
    #     edges[0].append(edge)
    #     edges[1].append(edge)

    return edges
# def sort_time(ground_truth, idx_map, rev_time,edges):
#     time_result=dict()
#
#     for it, r_id in enumerate(ground_truth.keys()):
#         edges[0].append(idx_map[r_id])
#         edges[1].append(idx_map[r_id[0]])
#         if rev_time[r_id][1]:



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.nfeat = nfeat
        self.gc1 = GCNConv(nfeat, nhid,add_self_loops=False,bias=False,normalize=False)
        self.gc2 = GCNConv(nhid, nclass,add_self_loops=False,bias=False,normalize=False)
        self.dropout = dropout
        self.linear_r = nn.Linear(5, nfeat)
        self.linear_p = nn.Linear(6, nfeat)
        self.linear_u = nn.Linear(7, nfeat)

    def forward(self, inputx, adj, nums):
        x = torch.zeros(len(inputx), self.nfeat)
        #        for it, k in enumerate(inputx):
        #            if len(k) == 5:
        #                x[it] = self.linear_r(torch.FloatTensor(k))
        #            elif len(k) == 6:
        #                x[it] = self.linear_p(torch.FloatTensor(k))
        #            else:
        #                x[it] = self.linear_u(torch.FloatTensor(k))
        x[:nums[0][0]] = self.linear_r(torch.FloatTensor(inputx[:nums[0][0]]))
        x[nums[0][0]:nums[0][1]] = self.linear_u(torch.FloatTensor(inputx[nums[0][0]:nums[0][1]]))
        if nums[1][0] != nums[1][1]:
            x[nums[0][1]:nums[1][0]] = self.linear_r(torch.FloatTensor(inputx[nums[0][1]:nums[1][0]]))
            x[nums[1][0]:nums[1][1]] = self.linear_u(torch.FloatTensor(inputx[nums[1][0]:nums[1][1]]))
        if nums[2][0] != nums[2][1]:
            x[nums[1][1]:nums[2][0]] = self.linear_r(torch.FloatTensor(inputx[nums[1][1]:nums[2][0]]))
            x[nums[2][0]:nums[2][1]] = self.linear_u(torch.FloatTensor(inputx[nums[2][0]:nums[2][1]]))
        x[nums[2][1]:] = self.linear_p(torch.FloatTensor(inputx[nums[2][1]:]))

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return x

    def output_layer(self, inputx, adj, is_cuda):
        x = torch.zeros(len(inputx), self.nfeat)
        for it, k in enumerate(inputx):
            k = k.type(torch.FloatTensor)
            if is_cuda:
                k = k.cuda()
            if len(k) == 5:
                x[it] = self.linear_r(k)
            elif len(k) == 6:
                x[it] = self.linear_p(k)
            else:
                x[it] = self.linear_u(k)
        # x[:nums[0][0]] = self.linear_r(torch.FloatTensor(inputx[:nums[0][0]]))
        # x[nums[0][0]:nums[0][1]] = self.linear_u(torch.FloatTensor(inputx[nums[0][0]:nums[0][1]]))
        # if nums[1][0] != nums[1][1]:
        #    x[nums[0][1]:nums[1][0]] = self.linear_r(torch.FloatTensor(inputx[nums[0][1]:nums[1][0]]))
        #    x[nums[1][0]:nums[1][1]] = self.linear_u(torch.FloatTensor(inputx[nums[1][0]:nums[1][1]]))
        # if nums[2][0] != nums[2][1]:
        #    x[nums[1][1]:nums[2][0]] = self.linear_r(torch.FloatTensor(inputx[nums[1][1]:nums[2][0]]))
        #    x[nums[2][0]:nums[2][1]] = self.linear_u(torch.FloatTensor(inputx[nums[2][0]:nums[2][1]]))
        # x[nums[2][1]:] = self.linear_p(torch.FloatTensor(inputx[nums[2][1]:]))
        if is_cuda:
            x = x.cuda()
        return F.relu(self.gc1(x, adj))

    def feature(self, inputx, nums):
        x = torch.zeros(len(inputx), self.nfeat)
        #        for it, k in enumerate(inputx):
        #            if len(k) == 5:
        #                x[it] = self.linear_r(torch.FloatTensor(k))
        #            elif len(k) == 6:
        #                x[it] = self.linear_p(torch.FloatTensor(k))
        #            else:
        #                x[it] = self.linear_u(torch.FloatTensor(k))
        x[:nums[0][0]] = self.linear_r(torch.FloatTensor(inputx[:nums[0][0]]))
        x[nums[0][0]:nums[0][1]] = self.linear_u(torch.FloatTensor(inputx[nums[0][0]:nums[0][1]]))
        if nums[1][0] != nums[1][1]:
            x[nums[0][1]:nums[1][0]] = self.linear_r(torch.FloatTensor(inputx[nums[0][1]:nums[1][0]]))
            x[nums[1][0]:nums[1][1]] = self.linear_u(torch.FloatTensor(inputx[nums[1][0]:nums[1][1]]))
        if nums[2][0] != nums[2][1]:
            x[nums[1][1]:nums[2][0]] = self.linear_r(torch.FloatTensor(inputx[nums[1][1]:nums[2][0]]))
            x[nums[2][0]:nums[2][1]] = self.linear_u(torch.FloatTensor(inputx[nums[2][0]:nums[2][1]]))
        x[nums[2][1]:] = self.linear_p(torch.FloatTensor(inputx[nums[2][1]:]))
        return x

    def back(self, features, adj):
        x_0 = self.gc1(features, adj)
        x_1 = F.relu(x_0)
        x_1 = F.dropout(x_1, self.dropout, training=self.training)
        x = self.gc2(x_1, adj)
        return (x_0, x_1, x)
def from_edge_findpaths(edge_list,graph):
    result=[]
    for edge in edge_list:
        middlenode=edge[0]
        targetnode=edge[1]
        for neighbor in graph[middlenode]:
            path = []
            path.append(neighbor)
            path.append(middlenode)
            path.append(targetnode)
            if path not in result:
                result.append(path)

            path = []
            path.append(targetnode)
            path.append(middlenode)
            path.append(neighbor)
            if path not in result:
                result.append(path)


        middlenode = edge[1]
        targetnode = edge[0]
        for neighbor in graph[middlenode]:
            path = []
            path.append(neighbor)
            path.append(middlenode)
            path.append(targetnode)
            if path not in result:
                result.append(path)

            path = []
            path.append(targetnode)
            path.append(middlenode)
            path.append(neighbor)
            if path not in result:
                result.append(path)
    return result







def from_edge_to_graph(edge_index):
    graph_dict=dict()

    for i in range(len(edge_index[0])):
        if edge_index[0][i] not in graph_dict.keys():
            graph_dict[edge_index[0][i]]=[edge_index[1][i]]
        else:
            if edge_index[1][i] not in graph_dict[edge_index[0][i]]:
                graph_dict[edge_index[0][i]].append(edge_index[1][i])
    return graph_dict

def rumor_construct_adj_matrix(edges_index,x):
    edges = []
    # print(ground_truth.keys())
    for idx,node in enumerate(edges_index[0]):
        edges.append((node,edges_index[1][idx]))


    edges = np.array(edges)
    adj = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(x, x),
                        dtype=np.float32)
    # for i in range(len(edges_index[0])):
    #     adj[edges_index[0][i],edges_index[1][i]]=edge_weight[i]
    adj=normalize(adj)
    return adj
def clear(edges):
    edge_clear=[]
    for idx,edge in enumerate(edges):
        # if idx%1000==0:
        #     # print('idx',idx)
        if [edge[0],edge[1]] not in edge_clear and [edge[1],edge[0]] not in edge_clear:
            edge_clear.append([edge[0],edge[1]])
    return edge_clear
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
def test_path_contribution_layeredge(paths,adj_start,adj_end,addedgelist,relu_delta,relu_start,relu_end,x_tensor,W1,W2):
    XW1=torch.mm(x_tensor,W1)
    path_result_dict=dict()
    node_result_dict=dict()
    edge_result_dict_zong=dict()
    for edge in addedgelist:
        for layer in range(2):
            edge_key = str(edge[0]) + ',' + str(edge[1]) + ',' + str(layer)
            edge_result_dict_zong[edge_key] = np.zeros((adj_end.shape[0], W2.shape[1]))
            edge_key = str(edge[1]) + ',' + str(edge[0]) + ',' + str(layer)
            edge_result_dict_zong[edge_key] = np.zeros((adj_end.shape[0], W2.shape[1]))

    for path in paths:
        if ([path[0],path[1]] in addedgelist or  [path[1],path[0]] in addedgelist) and ([path[2],path[1]] in addedgelist or  [path[1],path[2]] in addedgelist):
            # print(adj_end[path[1],path[2]])


            if adj_end[path[1],path[2]]>=adj_start[path[1],path[2]]:
                f1=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)
                f2=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_end[path[1],path[0]])*relu_end[path[1]]*XW1[path[0]],0),W2)
                f3=f1+f2

                f4=f1
                f5=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_start[path[1],path[0]])*relu_start[path[1]]*XW1[path[0]],0),W2)

            else:
                # print('torch.mul',torch.mul(relu_delta[path[1]],XW1[path[0]]))
                # weight=(adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*torch.mul(relu_delta[path[1]],XW1[path[0]])
                # print(weight.shape)
                # print(W2.shape)
                # print('f1',torch.mm(torch.unsqueeze(weight,0),W2))
                f1=adj_end[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*torch.mul(relu_delta[path[1]],XW1[path[0]]),0),W2)
                f2=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) * torch.mm(torch.unsqueeze((adj_start[path[1],path[0]]) * relu_start[path[1]] * XW1[path[0]],0)
                    , W2)
                f3=f1+f2

                f4=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)
                f5=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_start[path[1],path[0]])*relu_start[path[1]]*XW1[path[0]],0),W2)

                # f1 = adj_start[path[1]][path[2]] * np.dot(
                #     (adj_end[path[1]][path[0]] - adj_start[path[1]][path[0]]) * relu_delta[path[1]] * XW1[path[0]], W2)
                # f2 = (adj_end[path[1]][path[2]] - adj_start[path[1]][path[2]]) * np.dot(
                #     (adj_end[path[1]][path[0]]) * relu_end[path[1]] * XW1[path[0]], W2)
                # f3 = f1 + f2
                #
                # f4 = f1
                # f5 = (adj_end[path[1]][path[2]] - adj_start[path[1]][path[2]]) * np.dot(
                #     (adj_start[path[1]][path[0]]) * relu_start[path[1]] * XW1[path[0]], W2)

            # print('f1',f1)
            # print('f2', f2)
            # print('f3',f3)
            f3=torch.squeeze(f3,0)
            f4 = torch.squeeze(f4, 0)
            f5 = torch.squeeze(f5, 0)
            f3=f3.detach().numpy()
            f4=f4.detach().numpy()
            f5=f5.detach().numpy()

            contribution_edge_1 = 0.5 * (f3 - f5 + f4)
            contribution_edge_2 = 0.5 * (f3 - f4 + f5)

            # print('contribution_edge_1',contribution_edge_1)
            # print('contribution_edge_1.shape', contribution_edge_1.shape)

            p_key = str(path[0]) + ',' + str(path[1]) + ',' + str(path[2])
            path_result_dict[p_key] = f3
            if path[2] not in node_result_dict.keys():
                node_result_dict[path[2]] = f3
            else:
                node_result_dict[path[2]] += f3

            edge_1 = str(path[0]) + ',' + str(path[1]) + ',' + str(0)
            #print('contribution_edge_1',contribution_edge_1)
            edge_result_dict_zong[edge_1][path[2]] += contribution_edge_1


            edge_2 = str(path[1]) + ',' + str(path[2]) + ',' + str(1)
            edge_result_dict_zong[edge_2][path[2]] += contribution_edge_2

            # if [path[0],path[1]] in addedgelist:
            #     edge_1=str(path[0])+','+str(path[1])
            # else:
            #     edge_1 = str(path[1]) + ',' + str(path[0])
            #
            # if edge_1 in edge_result_dict_zong.keys():
            #     edge_result_dict_zong[edge_1][path[2]]+=contribution_edge_1
            # else:
            #     edge_result_dict_zong[edge_1][path[2]]= contribution_edge_1
            #
            # if [path[2], path[1]] in addedgelist:
            #     edge_2 = str(path[2]) + ',' + str(path[1])
            # else:
            #     edge_2 = str(path[1]) + ',' + str(path[2])
            #
            # if edge_2 in edge_result_dict_zong.keys():
            #     edge_result_dict_zong[edge_2][path[2]] += contribution_edge_2
            # else:
            #     edge_result_dict_zong[edge_2][path[2]] = contribution_edge_2



        else:
            for i in range(len(path) - 1):
                # edge_key = str(path[i]) + ',' + str(path[i + 1]) +','+ str(i)
                if [path[i], path[i + 1]] in addedgelist:
                    edge_key = str(path[i]) + ',' + str(path[i + 1]) + ',' + str(i)
                elif [path[i + 1], path[i]] in addedgelist:
                    edge_key = str(path[i]) + ',' + str(path[i + 1]) + ',' + str(i)

            if adj_end[path[1],path[2]]>=adj_start[path[1],path[2]]:
                contribution=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)+ \
                             (adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_end[path[1],path[0]])*relu_end[path[1]]*XW1[path[0]],0),W2)

            else:
                contribution=adj_end[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)+ \
                             (adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) * torch.mm(
                    torch.unsqueeze((adj_start[path[1],path[0]]) * relu_start[path[1]] * XW1[path[0]],0), W2)






            # if edge_key in edge_result_dict_zong.keys():
            #     edge_result_dict_zong[edge_key][path[2]]+=contribution
            # else:
            #     edge_result_dict_zong[edge_key][path[2]]= contribution
            contribution=torch.squeeze(contribution,0)
            contribution=contribution.detach().numpy()

            edge_result_dict_zong[edge_key][path[2]] += contribution

            p_key = str(path[0]) + ',' + str(path[1]) + ',' + str(path[2])
            path_result_dict[p_key] = contribution
            if path[2] not in node_result_dict.keys():
                node_result_dict[path[2]] = contribution
            else:
                node_result_dict[path[2]] += contribution

    return path_result_dict,node_result_dict,edge_result_dict_zong

def test_path_contribution_layeredge_without_shapley(paths,adj_start,adj_end,addedgelist,relu_delta,relu_start,relu_end,x_tensor,W1,W2):
    XW1=torch.mm(x_tensor,W1)
    path_result_dict=dict()
    node_result_dict=dict()
    edge_result_dict_zong=dict()
    for edge in addedgelist:
        for layer in range(2):
            edge_key = str(edge[0]) + ',' + str(edge[1]) + ',' + str(layer)
            edge_result_dict_zong[edge_key] = np.zeros((adj_end.shape[0], W2.shape[1]))
            edge_key = str(edge[1]) + ',' + str(edge[0]) + ',' + str(layer)
            edge_result_dict_zong[edge_key] = np.zeros((adj_end.shape[0], W2.shape[1]))

    for path in paths:
        if ([path[0],path[1]] in addedgelist or  [path[1],path[0]] in addedgelist) and ([path[2],path[1]] in addedgelist or  [path[1],path[2]] in addedgelist):
            # print(adj_end[path[1],path[2]])


            if adj_end[path[1],path[2]]>=adj_start[path[1],path[2]]:
                f1=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)
                f2=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_end[path[1],path[0]])*relu_end[path[1]]*XW1[path[0]],0),W2)
                f3=f1+f2
            else:
                # print('torch.mul',torch.mul(relu_delta[path[1]],XW1[path[0]]))
                # weight=(adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*torch.mul(relu_delta[path[1]],XW1[path[0]])
                # print(weight.shape)
                # print(W2.shape)
                # print('f1',torch.mm(torch.unsqueeze(weight,0),W2))
                f1=adj_end[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*torch.mul(relu_delta[path[1]],XW1[path[0]]),0),W2)
                f2=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) * torch.mm(torch.unsqueeze((adj_start[path[1],path[0]]) * relu_start[path[1]] * XW1[path[0]],0)
                    , W2)
                f3=f1+f2


            f3=torch.squeeze(f3,0)

            f3=f3.detach().numpy()


            contribution_edge_1 = 0.5 * f3
            contribution_edge_2 = 0.5 * f3

            # print('contribution_edge_1',contribution_edge_1)
            # print('contribution_edge_1.shape', contribution_edge_1.shape)

            p_key = str(path[0]) + ',' + str(path[1]) + ',' + str(path[2])
            path_result_dict[p_key] = f3
            if path[2] not in node_result_dict.keys():
                node_result_dict[path[2]] = f3
            else:
                node_result_dict[path[2]] += f3

            edge_1 = str(path[0]) + ',' + str(path[1]) + ',' + str(0)
            #print('contribution_edge_1',contribution_edge_1)
            edge_result_dict_zong[edge_1][path[2]] += contribution_edge_1


            edge_2 = str(path[1]) + ',' + str(path[2]) + ',' + str(1)
            edge_result_dict_zong[edge_2][path[2]] += contribution_edge_2

            # if [path[0],path[1]] in addedgelist:
            #     edge_1=str(path[0])+','+str(path[1])
            # else:
            #     edge_1 = str(path[1]) + ',' + str(path[0])
            #
            # if edge_1 in edge_result_dict_zong.keys():
            #     edge_result_dict_zong[edge_1][path[2]]+=contribution_edge_1
            # else:
            #     edge_result_dict_zong[edge_1][path[2]]= contribution_edge_1
            #
            # if [path[2], path[1]] in addedgelist:
            #     edge_2 = str(path[2]) + ',' + str(path[1])
            # else:
            #     edge_2 = str(path[1]) + ',' + str(path[2])
            #
            # if edge_2 in edge_result_dict_zong.keys():
            #     edge_result_dict_zong[edge_2][path[2]] += contribution_edge_2
            # else:
            #     edge_result_dict_zong[edge_2][path[2]] = contribution_edge_2



        else:
            for i in range(len(path) - 1):
                # edge_key = str(path[i]) + ',' + str(path[i + 1]) +','+ str(i)
                if [path[i], path[i + 1]] in addedgelist:
                    edge_key = str(path[i]) + ',' + str(path[i + 1]) + ',' + str(i)
                elif [path[i + 1], path[i]] in addedgelist:
                    edge_key = str(path[i]) + ',' + str(path[i + 1]) + ',' + str(i)

            if adj_end[path[1],path[2]]>=adj_start[path[1],path[2]]:
                contribution=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)+ \
                             (adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_end[path[1],path[0]])*relu_end[path[1]]*XW1[path[0]],0),W2)

            else:
                contribution=adj_end[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)+ \
                             (adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) * torch.mm(
                    torch.unsqueeze((adj_start[path[1],path[0]]) * relu_start[path[1]] * XW1[path[0]],0), W2)






            # if edge_key in edge_result_dict_zong.keys():
            #     edge_result_dict_zong[edge_key][path[2]]+=contribution
            # else:
            #     edge_result_dict_zong[edge_key][path[2]]= contribution
            contribution=torch.squeeze(contribution,0)
            contribution=contribution.detach().numpy()

            edge_result_dict_zong[edge_key][path[2]] += contribution

            p_key = str(path[0]) + ',' + str(path[1]) + ',' + str(path[2])
            path_result_dict[p_key] = contribution
            if path[2] not in node_result_dict.keys():
                node_result_dict[path[2]] = contribution
            else:
                node_result_dict[path[2]] += contribution

    return path_result_dict,node_result_dict,edge_result_dict_zong


def test_path_contribution_edge(paths,adj_start,adj_end,addedgelist,relu_delta,relu_start,relu_end,x_tensor,W1,W2):
    XW1=torch.mm(x_tensor,W1)
    path_result_dict=dict()
    node_result_dict=dict()
    edge_result_dict_zong=dict()
    for edge in addedgelist:
        edge_key=str(edge[0])+','+str(edge[1])
        edge_result_dict_zong[edge_key]=np.zeros((adj_end.shape[0],W2.shape[1]))


    for path in paths:
        if ([path[0],path[1]] in addedgelist or  [path[1],path[0]] in addedgelist) and ([path[2],path[1]] in addedgelist or  [path[1],path[2]] in addedgelist):
            # print(adj_end[path[1],path[2]])


            if adj_end[path[1],path[2]]>=adj_start[path[1],path[2]]:
                f1=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)
                f2=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_end[path[1],path[0]])*relu_end[path[1]]*XW1[path[0]],0),W2)
                f3=f1+f2

                f4=f1
                f5=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_start[path[1],path[0]])*relu_start[path[1]]*XW1[path[0]],0),W2)

            else:
                # print('torch.mul',torch.mul(relu_delta[path[1]],XW1[path[0]]))
                # weight=(adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*torch.mul(relu_delta[path[1]],XW1[path[0]])
                # print(weight.shape)
                # print(W2.shape)
                # print('f1',torch.mm(torch.unsqueeze(weight,0),W2))
                f1=adj_end[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*torch.mul(relu_delta[path[1]],XW1[path[0]]),0),W2)
                f2=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) * torch.mm(torch.unsqueeze((adj_start[path[1],path[0]]) * relu_start[path[1]] * XW1[path[0]],0)
                    , W2)
                f3=f1+f2

                f4=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)
                f5=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_start[path[1],path[0]])*relu_start[path[1]]*XW1[path[0]],0),W2)

                # f1 = adj_start[path[1]][path[2]] * np.dot(
                #     (adj_end[path[1]][path[0]] - adj_start[path[1]][path[0]]) * relu_delta[path[1]] * XW1[path[0]], W2)
                # f2 = (adj_end[path[1]][path[2]] - adj_start[path[1]][path[2]]) * np.dot(
                #     (adj_end[path[1]][path[0]]) * relu_end[path[1]] * XW1[path[0]], W2)
                # f3 = f1 + f2
                #
                # f4 = f1
                # f5 = (adj_end[path[1]][path[2]] - adj_start[path[1]][path[2]]) * np.dot(
                #     (adj_start[path[1]][path[0]]) * relu_start[path[1]] * XW1[path[0]], W2)

            # print('f1',f1)
            # print('f2', f2)
            # print('f3',f3)
            f3=torch.squeeze(f3,0)
            f4 = torch.squeeze(f4, 0)
            f5 = torch.squeeze(f5, 0)
            f3=f3.detach().numpy()
            f4=f4.detach().numpy()
            f5=f5.detach().numpy()

            contribution_edge_1 = 0.5 * (f3 - f5 + f4)
            contribution_edge_2 = 0.5 * (f3 - f4 + f5)

            # print('contribution_edge_1',contribution_edge_1)
            # print('contribution_edge_1.shape', contribution_edge_1.shape)

            p_key = str(path[0]) + ',' + str(path[1]) + ',' + str(path[2])
            path_result_dict[p_key] = f3
            if path[2] not in node_result_dict.keys():
                node_result_dict[path[2]] = f3
            else:
                node_result_dict[path[2]] += f3


            if [path[0],path[1]] in addedgelist:
                edge_1=str(path[0])+','+str(path[1])
            else:
                edge_1 = str(path[1]) + ',' + str(path[0])

            if edge_1 in edge_result_dict_zong.keys():
                edge_result_dict_zong[edge_1][path[2]]+=contribution_edge_1
            else:
                edge_result_dict_zong[edge_1][path[2]]= contribution_edge_1

            if [path[2], path[1]] in addedgelist:
                edge_2 = str(path[2]) + ',' + str(path[1])
            else:
                edge_2 = str(path[1]) + ',' + str(path[2])

            if edge_2 in edge_result_dict_zong.keys():
                edge_result_dict_zong[edge_2][path[2]] += contribution_edge_2
            else:
                edge_result_dict_zong[edge_2][path[2]] = contribution_edge_2



        else:
            for i in range(len(path) - 1):
                # edge_key = str(path[i]) + ',' + str(path[i + 1]) +','+ str(i)
                if [path[i], path[i + 1]] in addedgelist:
                    edge_key = str(path[i]) + ',' + str(path[i + 1])
                elif [path[i + 1], path[i]] in addedgelist:
                    edge_key = str(path[i + 1]) + ',' + str(path[i])

            if adj_end[path[1],path[2]]>=adj_start[path[1],path[2]]:
                contribution=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)+ \
                             (adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_end[path[1],path[0]])*relu_end[path[1]]*XW1[path[0]],0),W2)

            else:
                contribution=adj_end[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)+ \
                             (adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) * torch.mm(
                    torch.unsqueeze((adj_start[path[1],path[0]]) * relu_start[path[1]] * XW1[path[0]],0), W2)






            # if edge_key in edge_result_dict_zong.keys():
            #     edge_result_dict_zong[edge_key][path[2]]+=contribution
            # else:
            #     edge_result_dict_zong[edge_key][path[2]]= contribution
            contribution=torch.squeeze(contribution,0)
            contribution=contribution.detach().numpy()

            if edge_key in edge_result_dict_zong.keys():
                edge_result_dict_zong[edge_key][path[2]]+=contribution
            else:
                edge_result_dict_zong[edge_key][path[2]]= contribution

            p_key = str(path[0]) + ',' + str(path[1]) + ',' + str(path[2])
            path_result_dict[p_key] = contribution
            if path[2] not in node_result_dict.keys():
                node_result_dict[path[2]] = contribution
            else:
                node_result_dict[path[2]] += contribution

    return path_result_dict,node_result_dict,edge_result_dict_zong
def test_path_contribution_edge_without_shapley(paths,adj_start,adj_end,addedgelist,relu_delta,relu_start,relu_end,x_tensor,W1,W2):
    XW1=torch.mm(x_tensor,W1)
    path_result_dict=dict()
    node_result_dict=dict()
    edge_result_dict_zong=dict()
    for edge in addedgelist:
        edge_key=str(edge[0])+','+str(edge[1])
        edge_result_dict_zong[edge_key]=np.zeros((adj_end.shape[0],W2.shape[1]))


    for path in paths:
        if ([path[0],path[1]] in addedgelist or  [path[1],path[0]] in addedgelist) and ([path[2],path[1]] in addedgelist or  [path[1],path[2]] in addedgelist):
            # print(adj_end[path[1],path[2]])


            if adj_end[path[1],path[2]]>=adj_start[path[1],path[2]]:
                f1=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)
                f2=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_end[path[1],path[0]])*relu_end[path[1]]*XW1[path[0]],0),W2)
                f3=f1+f2


            else:
                # print('torch.mul',torch.mul(relu_delta[path[1]],XW1[path[0]]))
                # weight=(adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*torch.mul(relu_delta[path[1]],XW1[path[0]])
                # print(weight.shape)
                # print(W2.shape)
                # print('f1',torch.mm(torch.unsqueeze(weight,0),W2))
                f1=adj_end[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*torch.mul(relu_delta[path[1]],XW1[path[0]]),0),W2)
                f2=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) * torch.mm(torch.unsqueeze((adj_start[path[1],path[0]]) * relu_start[path[1]] * XW1[path[0]],0)
                    , W2)
                f3=f1+f2



            # print('f1',f1)
            # print('f2', f2)
            # print('f3',f3)
            f3=torch.squeeze(f3,0)

            f3=f3.detach().numpy()


            contribution_edge_1 = 0.5 * f3
            contribution_edge_2 = 0.5 * f3

            # print('contribution_edge_1',contribution_edge_1)
            # print('contribution_edge_1.shape', contribution_edge_1.shape)

            p_key = str(path[0]) + ',' + str(path[1]) + ',' + str(path[2])
            path_result_dict[p_key] = f3
            if path[2] not in node_result_dict.keys():
                node_result_dict[path[2]] = f3
            else:
                node_result_dict[path[2]] += f3


            if [path[0],path[1]] in addedgelist:
                edge_1=str(path[0])+','+str(path[1])
            else:
                edge_1 = str(path[1]) + ',' + str(path[0])

            if edge_1 in edge_result_dict_zong.keys():
                edge_result_dict_zong[edge_1][path[2]]+=contribution_edge_1
            else:
                edge_result_dict_zong[edge_1][path[2]]= contribution_edge_1

            if [path[2], path[1]] in addedgelist:
                edge_2 = str(path[2]) + ',' + str(path[1])
            else:
                edge_2 = str(path[1]) + ',' + str(path[2])

            if edge_2 in edge_result_dict_zong.keys():
                edge_result_dict_zong[edge_2][path[2]] += contribution_edge_2
            else:
                edge_result_dict_zong[edge_2][path[2]] = contribution_edge_2



        else:
            for i in range(len(path) - 1):
                # edge_key = str(path[i]) + ',' + str(path[i + 1]) +','+ str(i)
                if [path[i], path[i + 1]] in addedgelist:
                    edge_key = str(path[i]) + ',' + str(path[i + 1])
                elif [path[i + 1], path[i]] in addedgelist:
                    edge_key = str(path[i + 1]) + ',' + str(path[i])

            if adj_end[path[1],path[2]]>=adj_start[path[1],path[2]]:
                contribution=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)+ \
                             (adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_end[path[1],path[0]])*relu_end[path[1]]*XW1[path[0]],0),W2)

            else:
                contribution=adj_end[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)+ \
                             (adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) * torch.mm(
                    torch.unsqueeze((adj_start[path[1],path[0]]) * relu_start[path[1]] * XW1[path[0]],0), W2)

            contribution=torch.squeeze(contribution,0)
            contribution=contribution.detach().numpy()

            if edge_key in edge_result_dict_zong.keys():
                edge_result_dict_zong[edge_key][path[2]]+=contribution
            else:
                edge_result_dict_zong[edge_key][path[2]]= contribution

            p_key = str(path[0]) + ',' + str(path[1]) + ',' + str(path[2])
            path_result_dict[p_key] = contribution
            if path[2] not in node_result_dict.keys():
                node_result_dict[path[2]] = contribution
            else:
                node_result_dict[path[2]] += contribution

    return path_result_dict,node_result_dict,edge_result_dict_zong

def map_target(result_dict,target_node):
    final_dict=dict()
    for key,value in result_dict.items():
        final_dict[key]=value[target_node]
    return final_dict
def KL_divergence(P, Q):
    # Input P and Q would be vector (like messages or priors)
    # Will calculate the KL-divergence D-KL(P || Q) = sum~i ( P(i) * log(Q(i)/P(i)) )
    # Refer to Wikipedia https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    P = smooth(P)
    Q = smooth(Q)
    return sum(P * np.log(P / Q))
def main_con(select_number_path,goal,edge_result_dict, edgelist,old_tensor,new_tensor): #convex

    edge_selected= cvx.Variable(len(edgelist),integer=True)

    tmp_logits = copy.deepcopy(old_tensor)

    # print('edge_result_dict',edge_result_dict)
    # print('edgelist',edgelist)

    for i in range(len(edgelist)):
        add_matrix = np.array(
            edge_result_dict[str(edgelist[i][0]) + ',' + str(edgelist[i][1]) + ',' + str(edgelist[i][2])])
        tmp_logits = tmp_logits + edge_selected[i] * add_matrix

    # print(old_tensor.shape)
    # print('tmp_logits ',tmp_logits)

    new_prob=softmax(new_tensor)







    d=0
    for i in range(0,2):
        d=d+tmp_logits[i]*new_prob[i]
    # e=0
    # for i in range(0,ma.shape[1]):
    #     # e=e+cvx.atoms.exp(c[i])
    #     e = e + cvx.atoms.exp(y[i])
    # print('e',e)
    # print('e.shape',e.shape)
    #
    #
    #
    #
    # # f=sum(c*softmax(Hnew[layernumbers*2-1][goal]))
    # # print(c*softmax(Hnew[layernumbers*2-1][goal]).shape)
    # # print(cvx.atoms.log_sum_exp(c).shape)
    objective = cvx.Minimize(-d+cvx.atoms.log_sum_exp(tmp_logits))
    constraints = [sum(edge_selected)== select_number_path]

    for i in range(0,len(edgelist)):
        constraints.append(0 <= edge_selected[i])
        constraints.append(edge_selected[i] <= 1)
    # print(constraints)
    prob = cvx.Problem(objective, constraints)
    #
    prob.solve(solver='MOSEK') #solver='SCS' CVXOPT MOSEKMOSEK
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    # print('x.value',x.value)  # A numpy ndarray.**
    edge_res = []
    # group1_res = m.getAttr(group1_selected)
    # print('group0_res =', group0_res)

    for i in range(len(edgelist)):
        edge_res.append(
            edge_selected[i].value)

    # print('edge_res', edge_res)

    # result0 = [i for i, x in enumerate(edge_res) if abs(x - 1) < 1e-4]
    # print('result0',result0)

    sorted_id = sorted(range(len(edge_res)), key=lambda k: edge_res[k], reverse=True)

    # print('sorted_id',sorted_id)

    select_edges_list = []
    for i in range(select_number_path):
        # print(edge_res[sorted_id[i]])
        select_edges_list.append([edgelist[sorted_id[i]][0], edgelist[sorted_id[i]][1], edgelist[sorted_id[i]][2]])

    # print('select_edges_list', select_edges_list)

    return select_edges_list
def smooth(arr, eps=1e-5):
    if 0 in arr:
        return abs(arr - eps)
    else:
        return arr
def from_layeredges_to_evaulate(select_layeredges_list,edges_weight_old,edges_old,edges_old_dict,adj_old,adj_new):
    evaluate_edge_weight1 = copy.deepcopy(edges_weight_old.tolist())
    evaluate_edge_weight2 = copy.deepcopy(edges_weight_old.tolist())

    evaluate_edge_index1 = copy.deepcopy(edges_old.tolist())
    evaluate_edge_index2 = copy.deepcopy(edges_old.tolist())

    for layeredge in select_layeredges_list:
        # print('layeredge',layeredge)
        if layeredge[2] == 0:
            if adj_old[layeredge[0], layeredge[1]] != 0:
                value = edges_old_dict[str(layeredge[0]) + ',' + str(layeredge[1])]
                evaluate_edge_weight1[value] = adj_new[layeredge[0], layeredge[1]]
                # print('exist',layeredge)
            else:
                evaluate_edge_index1[0].append(layeredge[0])
                evaluate_edge_index1[1].append(layeredge[1])
                evaluate_edge_weight1.append(adj_new[layeredge[0], layeredge[1]])
                # print('add',layeredge)
                # print('adj_new[layeredge[0],layeredge[1]]',adj_new[layeredge[0],layeredge[1]])
        elif layeredge[2] == 1:
            if adj_old[layeredge[0], layeredge[1]] != 0:
                value = edges_old_dict[str(layeredge[0]) + ',' + str(layeredge[1])]
                evaluate_edge_weight2[value] = adj_new[layeredge[0], layeredge[1]]
                # print('exist')
            else:
                evaluate_edge_index2[0].append(layeredge[0])
                evaluate_edge_index2[1].append(layeredge[1])
                evaluate_edge_weight2.append(adj_new[layeredge[0], layeredge[1]])
                # print('add')
                # print('adj_new[layeredge[0],layeredge[1]]',adj_new[layeredge[0],layeredge[1]])
    return evaluate_edge_index1,evaluate_edge_index2,evaluate_edge_weight1,evaluate_edge_weight2
def from_edges_to_evaulate(select_edges_list,edges_weight_old,edges_old,edges_old_dict,adj_old,adj_new):
    evaluate_edge_weight = copy.deepcopy(edges_weight_old.tolist())
    evaluate_edge_index= copy.deepcopy(edges_old.tolist())

    for edge in select_edges_list:
        if edge[0] != edge[1]:
            if adj_old[edge[0], edge[1]] != 0:
                value1 = edges_old_dict[str(edge[0]) + ',' + str(edge[1])]
                evaluate_edge_weight[value1] = adj_new[edge[0], edge[1]]

                value2 = edges_old_dict[str(edge[1]) + ',' + str(edge[0])]
                evaluate_edge_weight[value2] = adj_new[edge[1], edge[0]]
            else:
                evaluate_edge_index[0].append(edge[0])
                evaluate_edge_index[1].append(edge[1])
                evaluate_edge_weight.append(adj_new[edge[0], edge[1]])

                evaluate_edge_index[0].append(edge[1])
                evaluate_edge_index[1].append(edge[0])
                evaluate_edge_weight.append(adj_new[edge[1], edge[0]])


        else:
            if adj_old[edge[0], edge[1]] != 0:
                value1 = edges_old_dict[str(edge[0]) + ',' + str(edge[1])]
                evaluate_edge_weight[value1] = adj_new[edge[0], edge[1]]
            else:
                evaluate_edge_index[0].append(edge[0])
                evaluate_edge_index[1].append(edge[1])
                evaluate_edge_weight.append(adj_new[edge[0], edge[1]])

    return evaluate_edge_index,evaluate_edge_weight

def from_edges_to_evaulate_2(select_edges_list,sub_mapping,edges_weight_old,edges_old,edges_old_dict,adj_old,adj_new):
    evaluate_edge_weight = copy.deepcopy(edges_weight_old.tolist())
    evaluate_edge_index= copy.deepcopy(edges_old.tolist())

    a=len(sub_mapping)
    count=0

    for edge in select_edges_list:
        if edge[0] not in sub_mapping.keys():
            sub_mapping[edge[0]]=a+count
            count+=1


        if edge[0] != edge[1]:
            if adj_old[edge[0], edge[1]] != 0:
                value1 = edges_old_dict[str(edge[0]) + ',' + str(edge[1])]
                evaluate_edge_weight[value1] = adj_new[edge[0], edge[1]]

                value2 = edges_old_dict[str(edge[1]) + ',' + str(edge[0])]
                evaluate_edge_weight[value2] = adj_new[edge[1], edge[0]]
            else:
                evaluate_edge_index[0].append(edge[0])
                evaluate_edge_index[1].append(edge[1])
                evaluate_edge_weight.append(adj_new[edge[0], edge[1]])

                evaluate_edge_index[0].append(edge[1])
                evaluate_edge_index[1].append(edge[0])
                evaluate_edge_weight.append(adj_new[edge[1], edge[0]])


        else:
            if adj_old[edge[0], edge[1]] != 0:
                value1 = edges_old_dict[str(edge[0]) + ',' + str(edge[1])]
                evaluate_edge_weight[value1] = adj_new[edge[0], edge[1]]
            else:
                evaluate_edge_index[0].append(edge[0])
                evaluate_edge_index[1].append(edge[1])
                evaluate_edge_weight.append(adj_new[edge[0], edge[1]])

    return evaluate_edge_index,evaluate_edge_weight
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def main_con_edge(select_number_path,edge_result_dict, edgelist,old_tensor,new_tensor): #convex

    edge_selected= cvx.Variable(len(edgelist),integer=True)

    tmp_logits = copy.deepcopy(old_tensor)

    # print('edge_result_dict',edge_result_dict)
    # print('edgelist',edgelist)

    for i in range(len(edgelist)):
        add_matrix = np.array(
            edge_result_dict[str(edgelist[i][0]) + ',' + str(edgelist[i][1])])
        tmp_logits = tmp_logits + edge_selected[i] * add_matrix

    # print(old_tensor.shape)
    # print('tmp_logits ',tmp_logits)

    new_prob=softmax(new_tensor)
    d=0
    for i in range(0,2):
        d=d+tmp_logits[i]*new_prob[i]
    # e=0
    # for i in range(0,ma.shape[1]):
    #     # e=e+cvx.atoms.exp(c[i])
    #     e = e + cvx.atoms.exp(y[i])
    # print('e',e)
    # print('e.shape',e.shape)
    #
    #
    #
    #
    # # f=sum(c*softmax(Hnew[layernumbers*2-1][goal]))
    # # print(c*softmax(Hnew[layernumbers*2-1][goal]).shape)
    # # print(cvx.atoms.log_sum_exp(c).shape)
    objective = cvx.Minimize(-d+cvx.atoms.log_sum_exp(tmp_logits))
    constraints = [sum(edge_selected)== select_number_path]

    for i in range(0,len(edgelist)):
        constraints.append(0 <= edge_selected[i])
        constraints.append(edge_selected[i] <= 1)
    # print(constraints)
    prob = cvx.Problem(objective, constraints)
    #
    prob.solve(solver='MOSEK') #solver='SCS' CVXOPT MOSEKMOSEK
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    # print('x.value',x.value)  # A numpy ndarray.**
    edge_res = []
    # group1_res = m.getAttr(group1_selected)
    # print('group0_res =', group0_res)

    for i in range(len(edgelist)):
        edge_res.append(
            edge_selected[i].value)

    # print('edge_res', edge_res)

    # result0 = [i for i, x in enumerate(edge_res) if abs(x - 1) < 1e-4]
    # print('result0',result0)

    sorted_id = sorted(range(len(edge_res)), key=lambda k: edge_res[k], reverse=True)

    select_edges_list = []
    for i in range(select_number_path):
        select_edges_list.append([edgelist[sorted_id[i]][0], edgelist[sorted_id[i]][1]])

        # print('edge contribution',edge_res[sorted_id[i]])

    # print('select_edges_list', select_edges_list)

    return select_edges_list




class GCN_test(torch.nn.Module):
    def __init__(self, nfeat,nhid, nclass,dropout):
        super(GCN_test, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid,normalize=False,add_self_loops=False,bias=False)
        self.conv2 = GCNConv(nhid, nclass,normalize=False,add_self_loops=False,bias=False)

        # self.conv2=GCNConv(nhid1, nhid2 )
        # self.conv3 = GCNConv(nhid2, dataset.num_classes)
        # self.conv4 = GCNConv(nhid3, dataset.num_classes)
        self.dropout = dropout





    def forward(self, x, edge_index1,edge_index2,edge_weight1,edge_weight2):
        # print(self.conv1(x, edge_index))
        x = F.relu(self.conv1(x, edge_index1,edge_weight=edge_weight1))

        # x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv2(x, edge_index2,edge_weight=edge_weight2)

        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.conv3(x,edge_index)
        return x
        # return F.log_softmax(x, dim=1)

    def back(self, x, edge_index_1, edge_index_2, edgeweight1, edgeweight2):
        x_0 = self.conv1(x, edge_index_1, edge_weight=edgeweight1)
        x_1 = F.relu(x_0)
        return (x_0, x_1)


class GCN_test_v2(torch.nn.Module):
    def __init__(self, nfeat,nhid, nclass,dropout):
        super(GCN_test_v2, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid,normalize=False,add_self_loops=False,bias=False)
        self.conv2 = GCNConv(nhid, nclass,normalize=False,add_self_loops=False,bias=False)

        # self.conv2=GCNConv(nhid1, nhid2 )
        # self.conv3 = GCNConv(nhid2, dataset.num_classes)
        # self.conv4 = GCNConv(nhid3, dataset.num_classes)
        self.dropout = dropout





    def forward(self, x, edge_index,edge_weight):
        # print(self.conv1(x, edge_index))
        x = F.relu(self.conv1(x, edge_index,edge_weight=edge_weight))

        # x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv2(x, edge_index,edge_weight=edge_weight)

        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.conv3(x,edge_index)
        return x
        # return F.log_softmax(x, dim=1)
    def back(self, x, edge_index,edge_weight):
        x1 = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))

        # x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv2(x1, edge_index, edge_weight=edge_weight)

        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.conv3(x,edge_index)
        return x1,x


def difference_weight(edgeindex1,edgeindex2,adj_new,adj_old): #tensor

    changed_edgelist=[]
    for idx,node in enumerate(edgeindex1[0]):
        # print('node',node)
        # print('edgeindex1[1][idx]',edgeindex1[1][idx])
        row_index=node
        col_index=edgeindex1[1][idx]
        value_new=adj_new[row_index,col_index]
        value_old = adj_old[row_index, col_index]
        if abs(value_old-value_new)>1e-5 and [row_index,col_index] not in changed_edgelist and [col_index,row_index] not in changed_edgelist:
            changed_edgelist.append([row_index,col_index])

    for idx,node in enumerate(edgeindex2[0]):
        # print('node',node)
        # print('edgeindex1[1][idx]',edgeindex1[1][idx])
        row_index=node
        col_index=edgeindex2[1][idx]
        value_new=adj_new[row_index,col_index]
        value_old = adj_old[row_index, col_index]
        if abs(value_old-value_new)>1e-3 and [row_index,col_index] not in changed_edgelist and [col_index,row_index] not in changed_edgelist:
            changed_edgelist.append([row_index,col_index])

    return changed_edgelist





    # for idx,node in enumerate(edgeindex2[0]):
    #     edgedict2[node,edgeindex2[1][idx]]=idx
    # for key in edgedict1.keys():
    #     if key not in edgedict2.keys():
    #         in1_not2.append(key)
    # for key in edgedict2.keys():
    #     if key not in edgedict1.keys():
    #         in2_not1.append(key)
    # # print('in1_not2',in1_not2)
    # # print('in2_not1', in2_not1)
    # print(len(in1_not2))
    # print(len(in2_not1))
    # return in1_not2,in2_not1


