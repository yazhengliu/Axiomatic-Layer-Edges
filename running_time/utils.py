import numpy as np
import scipy.sparse as sp
import torch
import math
import cvxpy as cvx
import random
import copy
import time
from torch import Tensor
def rumor_construct_adj_matrix(edges_index,x):
    edges = []
    # print(ground_truth.keys())
    for idx,node in enumerate(edges_index[0]):
        edges.append((node,edges_index[1][idx]))


    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(x, x),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])
    adj=normalize(adj)
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
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  #转换为coo矩阵
    indices = torch.from_numpy( np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
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
def split_train_test(labels,train_ratio,val_ratio):
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
def clear(edges):
    edge_clear=[]
    for idx,edge in enumerate(edges):
        # if idx%1000==0:
        #     # print('idx',idx)
        if [edge[0],edge[1]] not in edge_clear and [edge[1],edge[0]] not in edge_clear:
            edge_clear.append([edge[0],edge[1]])
    return edge_clear
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def smooth(arr, eps=1e-5):
    if 0 in arr:
        return abs(arr - eps)
    else:
        return arr


def KL_divergence(P, Q):
    # Input P and Q would be vector (like messages or priors)
    # Will calculate the KL-divergence D-KL(P || Q) = sum~i ( P(i) * log(Q(i)/P(i)) )
    # Refer to Wikipedia https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    P = smooth(P)
    Q = smooth(Q)
    return sum(P * np.log(P / Q))
def dfs(start,index,end,graph,length,path=[],paths=[]):#找到起点到终点的全部路径
    path.append(index)
    if len(path)==length:
        if path[-1]==end:
            paths.append(path.copy())
            path.pop()
        else:
            path.pop()

    else:
        for item in graph[index]:
            # if item not in path:
                dfs(start,item,end,graph,length,path,paths)
        path.pop()
    return paths
def dfs2(start,index,graph,length,path=[],paths=[]):#找到起点长度定值的全部路径
    path.append(index)
    # print('index',index)
    # if length==0:
    #     return paths
    if len(path)==length:
        paths.append(path.copy())
        path.pop()
    else:
        for item in graph[index]:
            # if item not in path:
                dfs2(start,item,graph,length,path,paths)
        path.pop()

    return paths

def findnewpath(addedgelist,graph,layernumbers,goal):
    resultpath=[]
    for edge in addedgelist:
        # print(edge)
        if edge[0] == goal:
            pt5 = dfs2(edge[1], edge[1], graph, layernumbers, [], [])
            # print(pt5)
            for i1 in pt5:
                # print(i1)
                i1.pop(0)
                if [goal, edge[1]] + i1 not in resultpath:
                    resultpath.append([goal, edge[1]] + i1)

        if edge[1] == goal:
            pt6 = dfs2(edge[0], edge[0], graph, layernumbers, [], [])
            for i1 in pt6:
                i1.pop(0)
                if [goal, edge[0]] + i1 not in resultpath:
                    resultpath.append([goal, edge[0]] + i1)

        for i in range(0, layernumbers - 1):
            # print('i', i)
            pt1 = dfs(goal, goal, edge[0], graph, i + 2, [], [])
            pt2 = dfs2(edge[1], edge[1], graph, layernumbers - i - 1, [], [])
            # print('pt1', pt1)
            # print('pt2', pt2)
            if pt2 != [] or pt1 != []:
                for i1 in pt1:
                    for j1 in pt2:
                        # print(i1 + j1)
                        if i1 + j1 not in resultpath:
                            resultpath.append(i1 + j1)

            # print(edge[1])
            # print(i)
            pt3 = dfs(goal, goal, edge[1], graph, i + 2, [], [])
            pt4 = dfs2(edge[0], edge[0], graph, layernumbers - i - 1, [], [])
            # print('pt3', pt3)
            # print('pt4', pt4)
            if pt3 != [] or pt4 != []:
                for i1 in pt3:
                    for j1 in pt4:
                        # print(i1 + j1)
                        if i1 + j1 not in resultpath:
                            resultpath.append(i1 + j1)
    return resultpath
def reverse_paths(pathlist):
    result=[]
    for path in pathlist:
        path.reverse()
        result.append(path)
    return result

def test_path_contribution_layeredge(paths,adj_start,adj_end,addedgelist,relu_delta,relu_start,relu_end,x_tensor,W1,W2):
    XW1=torch.mm(x_tensor,W1)
    path_result_dict=dict()
    node_result_dict=dict()
    edge_result_dict_zong=dict()
    contribution_time=0
    shapley_time=0
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
                # print(torch.mm(adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],W2)

                start_time=time.time()

                f1=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)
                f2=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_end[path[1],path[0]])*relu_end[path[1]]*XW1[path[0]],0),W2)
                f3=f1+f2
                middle_time=time.time()
                f4=f1
                f5=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_start[path[1],path[0]])*relu_start[path[1]]*XW1[path[0]],0),W2)
                end_time=time.time()
                contribution_time+=middle_time-start_time
                shapley_time+=end_time-middle_time

            else:
                # print('torch.mul',torch.mul(relu_delta[path[1]],XW1[path[0]]))
                # weight=(adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*torch.mul(relu_delta[path[1]],XW1[path[0]])
                # print(weight.shape)
                # print(W2.shape)
                # print('f1',torch.mm(torch.unsqueeze(weight,0),W2))
                start_time = time.time()
                f1=adj_end[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*torch.mul(relu_delta[path[1]],XW1[path[0]]),0),W2)
                f2=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) * torch.mm(torch.unsqueeze((adj_start[path[1],path[0]]) * relu_start[path[1]] * XW1[path[0]],0)
                    , W2)
                f3=f1+f2
                middle_time = time.time()

                f4=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)
                f5=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_start[path[1],path[0]])*relu_start[path[1]]*XW1[path[0]],0),W2)
                end_time = time.time()

                contribution_time += middle_time - start_time
                shapley_time += end_time - middle_time

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
            if p_key not in path_result_dict.keys():
                path_result_dict[p_key] = f3
            else:
                print('1,false')


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
                start_time=time.time()
                contribution=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)+ \
                             (adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_end[path[1],path[0]])*relu_end[path[1]]*XW1[path[0]],0),W2)
                end_time=time.time()
                contribution_time+=end_time-start_time

            else:
                start_time = time.time()
                contribution=adj_end[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)+ \
                             (adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) * torch.mm(
                    torch.unsqueeze((adj_start[path[1],path[0]]) * relu_start[path[1]] * XW1[path[0]],0), W2)
                end_time = time.time()

                contribution_time += end_time - start_time

            contribution = torch.squeeze(contribution, 0)
            contribution = contribution.detach().numpy()






            if edge_key in edge_result_dict_zong.keys():
                edge_result_dict_zong[edge_key][path[2]]+=contribution
            else:
                edge_result_dict_zong[edge_key][path[2]]= contribution


            # edge_result_dict_zong[edge_key][path[2]] += contribution

            p_key = str(path[0]) + ',' + str(path[1]) + ',' + str(path[2])
            path_result_dict[p_key] = contribution

            # if p_key not in path_result_dict.keys():
            #     path_result_dict[p_key] = contribution
            # else:
            #     print('2,false')

            if path[2] not in node_result_dict.keys():
                node_result_dict[path[2]] = contribution
            else:
                node_result_dict[path[2]] += contribution

    return path_result_dict,node_result_dict,edge_result_dict_zong,contribution_time,shapley_time

def main_con(select_number_path,goal,edge_result_dict, edgelist,old_tensor,output_new,numclass): #convex

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

    new_prob=softmax(output_new[goal].detach().numpy())



    d=0
    for i in range(0,numclass):
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
    prob.solve(solver='MOSEK',warm_start=True,mosek_params = {'MSK_DPAR_OPTIMIZER_MAX_TIME':  500.0,
                                    }) #solver='SCS' CVXOPT MOSEKMOSEK
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
def map_target(result_dict,target_node):
    final_dict=dict()
    for key,value in result_dict.items():
        final_dict[key]=value[target_node]
    return final_dict

def from_layeredges_to_evaulate(select_layeredges_list,edges_weight_old,edges_old,edges_old_dict,adj_old,adj_new):
    evaluate_edge_weight1 = copy.deepcopy(edges_weight_old.tolist())
    evaluate_edge_weight2 = copy.deepcopy(edges_weight_old.tolist())

    evaluate_edge_index1 = copy.deepcopy(edges_old)
    evaluate_edge_index2 = copy.deepcopy(edges_old)

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
def subadj_map(subset,edge_index,adj):
    mapping = dict()
    for it, neighbor in enumerate(subset):
        mapping[neighbor.item()] = it

    flipped_mapping = {v: k for k, v in mapping.items()}

    map_edge_index=[[],[]]
    edge_weight=[]
    for i in range(len(edge_index[0])):
        map_edge_index[0].append(mapping[edge_index[0][i].item()])
        map_edge_index[1].append(mapping[edge_index[1][i].item()])
        edge_weight.append(adj[edge_index[0][i],edge_index[1][i].item()])


    return mapping,flipped_mapping,map_edge_index,edge_weight
def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))
def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)
    # print('node_idx',node_idx)
    # print('node_mask',node_mask)
    # print('edge_mask', edge_mask)
    # print(len(row))
    inv = None

    subsets = [node_idx]

    for _ in range(num_hops):
        # print(num_hops)
        node_mask.fill_(False)
        # print(node_mask)
        node_mask[subsets[-1]] = True
        # print(row)
        torch.index_select(node_mask, 0, row, out=edge_mask)
        # print(torch.index_select(node_mask, 0, row, out=edge_mask))
        subsets.append(col[edge_mask])
    # print(subsets)
    # print('lensubsets',len(subsets[3]))
    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    # print('inv',inv)
    # print('subset',subset)
    inv = inv[:node_idx.numel()]
    # print(subsets)
    # print('inv', inv)
    # print(node_idx.numel())

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]
    # print(edge_mask)
    # print(len(edge_mask))

    edge_index = edge_index[:, edge_mask]
    # print(len(edge_index[0]))
    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask
def subfeaturs(features,mapping):
    subarray = np.array(np.zeros((len(mapping), features.shape[1])))
    for i in range(len(mapping)):
        subarray[i]=features[mapping[i]]
    return subarray
def rumor_construct_adj_matrix_v2(edges_index,x,edge_weight):
    edges = []
    # print(ground_truth.keys())
    for idx,node in enumerate(edges_index[0]):
        edges.append((node,edges_index[1][idx]))


    edges = np.array(edges)
    adj = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(x, x),
                        dtype=np.float32)
    for i in range(len(edges_index[0])):
        adj[edges_index[0][i],edges_index[1][i]]=edge_weight[i]
    return adj