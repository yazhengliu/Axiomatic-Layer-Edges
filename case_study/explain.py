import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
import random
import json
import copy
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import os
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from torch_geometric.nn import GCN
from torch_geometric.utils import k_hop_subgraph
import torch
from torch_geometric.nn import GCNConv
from utils import rumor_construct_adj_matrix,matrixtodict,clear,difference_weight,findnewpath,reverse_paths,test_path_contribution_layeredge,\
    KL_divergence,softmax,main_con,from_layeredges_to_evaulate

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
def gen_parameters(model,x_tensor,edges_new,edges_old,edgeweight1,edgeweight2):

    model.eval()

    nonlinear_start_layer1, nonlinear_relu_start_layer1 = model.back(x_tensor, edges_old,
                                                                     edgeweight1, )
    nonlinear_end_layer1, nonlinear_relu_end_layer1 = model.back(x_tensor, edges_new,
                                                                 edgeweight2)

    # print('nonlinear_start_layer1',nonlinear_start_layer1)

    relu_delta = torch.where((nonlinear_end_layer1 - nonlinear_start_layer1) != 0,
                             (nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1) / (
                                     nonlinear_end_layer1 - nonlinear_start_layer1),
                             torch.zeros_like((nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1)))
    relu_end = torch.where((nonlinear_end_layer1) != 0, nonlinear_relu_end_layer1 / nonlinear_end_layer1,
                           torch.zeros_like(nonlinear_end_layer1))
    relu_start = torch.where((nonlinear_start_layer1) != 0, nonlinear_relu_start_layer1 / nonlinear_start_layer1,
                             torch.zeros_like(nonlinear_start_layer1))
    W1 = model.state_dict()['conv1.lin.weight'].t()
    W2 = model.state_dict()['conv2.lin.weight'].t()
    return W1,W2 ,relu_delta,relu_end,relu_start
if __name__=='__main__':
    changed_ratio = 0.9
    data_type = 'house'
    data_path = f'gen_{data_type}_data/{changed_ratio}'
    model = GCN(nfeat=11, hidden_channels=20, nclass=2)

    model_path=f'{data_type}_GCN_model_{changed_ratio}.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    layernumbers=2
    count=0
    total=0

    for target_node in range(1000):

        json_path = data_path + '/' + str(target_node) + '/' + 'positive.json'
        with open(json_path, 'r') as f:
            result = json.load(f)

        x, edge_index_old,edge_weight_old = result['x'], result['edge_index'],result['edge_weight']
        x = np.array(result['x'])
        x = torch.tensor(x)
        x = x.to(torch.float32)
        node_labels = result['y']
        # edge_index_old = torch.tensor(edge_index_old)

        json_path = data_path + '/' + str(target_node) + '/' + 'negative.json'
        with open(json_path, 'r') as f:
            result = json.load(f)
        edge_index_new = result['edge_index']
        edge_weight_new=result['edge_weight']


        in_motif_nodes_list = []
        notin_motif_nodes_list = []
        for i in range(x.shape[0]):
            if node_labels[i] == 0:
                notin_motif_nodes_list.append(i)
            else:
                in_motif_nodes_list.append(i)

        adj_old = rumor_construct_adj_matrix(edge_index_old, x.shape[0],edge_weight_old)
        adj_new = rumor_construct_adj_matrix(edge_index_new, x.shape[0],edge_weight_new)

        # print(adj_new)

        if (adj_old.todense() == adj_old.todense().T).all():
            print("adj_old是对称矩阵。")
        else:
            print("adj_old不是对称矩阵。")

        if (adj_new.todense() == adj_new.todense().T).all():
            print("adj_new是对称矩阵。")
        else:
            print("adj_new不是对称矩阵。")




        adj_new_nonzero = adj_old.nonzero()
        adj_old_nonzero = adj_new.nonzero()


        edges_old_dict = dict()
        for i in range(len(edge_index_old[0])):
            edges_old_dict[str(edge_index_old[0][i]) + ',' + str(edge_index_old[1][i])] = i

        edges_new_dict = dict()

        for i in range(len(edge_index_new[0])):
            edges_new_dict[str(edge_index_new[0][i]) + ',' + str(edge_index_new[1][i])] = i


        changededgelist = difference_weight(edge_index_new, edge_index_old, adj_new, adj_old)
        # print(addedgelist)
        changededgelist = clear(changededgelist)

        graph_old = matrixtodict(adj_old_nonzero)
        graph_new = matrixtodict(adj_new_nonzero)

        # print('graph_old',graph_old)

        graph_all = copy.deepcopy(graph_old)

        # print('graph_all', graph_all)
        for i in range(x.shape[0]):
            if i not in graph_all.keys():
                graph_all[i]=[]

        for edge in changededgelist:
            if edge[0] not in graph_all.keys():
                graph_all[edge[0]] = [edge[1]]
            else:
                if edge[1] not in graph_all[edge[0]]:
                    graph_all[edge[0]].append(edge[1])

            if edge[1] not in graph_all.keys():
                graph_all[edge[1]] = [edge[0]]
            else:
                if edge[0] not in graph_all[edge[1]]:
                    graph_all[edge[1]].append(edge[0])

        edges_old_dict_reverse = dict()
        for key, value in edges_old_dict.items():
            edges_old_dict_reverse[value] = key

        edges_new_dict_reverse = dict()
        for key, value in edges_new_dict.items():
            edges_new_dict_reverse[value] = key

        edge_weight_old = torch.tensor(edge_weight_old)
        edge_weight_old = edge_weight_old.to(torch.float32)

        edge_weight_new = torch.tensor(edge_weight_new)
        edge_weight_new = edge_weight_new.to(torch.float32)

        edge_index_old = torch.tensor(edge_index_old)
        edge_index_new = torch.tensor(edge_index_new)

        output_old = model.forward(x, edge_index_old, edge_weight_old).view(-1)
        output_new = model.forward(x, edge_index_new, edge_weight_new).view(-1)

        KL_original = KL_divergence(softmax(output_new.detach().numpy()),
                                    softmax(output_old.detach().numpy()))
        # print('G_new', softmax(output_new.detach().numpy()))
        # # print('G_ceshi_new',softmax(output_ceshi_new[0].detach().numpy()))
        # print('G_old', softmax(output_old.detach().numpy()))
        # # print('G_ceshi_old', softmax(output_ceshi_old[0].detach().numpy()))

        # print('KL_original', KL_original)
        if KL_original>0.01:
            print('KL_original',KL_original,target_node)

        print('output_old',output_old)
        print('output_new', output_new)

        predict_new_label = np.argmax(softmax(output_new.detach().numpy()))

        predict_old_label = np.argmax(softmax(output_old.detach().numpy()))


        logit_old = model.pre_forward(x, edge_index_old, edge_weight_old)

        logit_new = model.pre_forward(x, edge_index_new, edge_weight_new)

        if predict_old_label==1 and predict_new_label==0:
            W1, W2, relu_delta, relu_end, relu_start = gen_parameters(model, x, edge_index_new, edge_index_old,
                                                                      edge_weight_old, edge_weight_new)
            path_ceshi = []
            for change_node in range(x.shape[0]):
                path_ceshi = path_ceshi + findnewpath(changededgelist, graph_all, layernumbers, change_node)

            # print('path_ceshi', len(path_ceshi))
            target_path = reverse_paths(path_ceshi)

            layer_edge_list = []
            for edge in changededgelist:
                for layer in range(layernumbers):
                    if edge[0] != edge[1]:
                        layer_edge_list.append([edge[0], edge[1], layer])
                        layer_edge_list.append([edge[1], edge[0], layer])
                    else:
                        layer_edge_list.append([edge[0], edge[1], layer])

            print('len layer_edge_list', len(layer_edge_list))

            _, _, test_layeredge_result_nonlinear = test_path_contribution_layeredge(target_path,
                                                                                     adj_old,
                                                                                     adj_new,
                                                                                     changededgelist,
                                                                                     relu_delta,
                                                                                     relu_start,
                                                                                     relu_end, x, W1, W2)

            flag = True
            ceshi_edge_result = np.zeros((adj_old.shape[0], 2))

            # print(ceshi_edge_result)
            for key, value in test_layeredge_result_nonlinear.items():
                # print(key,value)
                ceshi_edge_result += value

            true_diff_logits_nonlinear = logit_new.detach().numpy() - logit_old.detach().numpy()
            # print('ceshi_edge_result', ceshi_edge_result)
            # print('diff',true_diff_logits_nonlinear)
            for i in range(adj_old.shape[0]):
                if true_diff_logits_nonlinear[i].any() != 0:
                    if np.all(abs(ceshi_edge_result[i] - true_diff_logits_nonlinear[i]) > 1e-4):
                        print('key', i, 'test', ceshi_edge_result[i], 'true', true_diff_logits_nonlinear[i])
                        flag = False
            print('layeredge flag', flag)

            # old_data = Data(x=x, edge_index=edge_index_old)
            #
            # G = to_networkx(old_data, to_undirected=True)
            #
            # # 绘制图的拓扑结构
            # plt.figure(figsize=(8, 8))
            # pos = nx.spring_layout(G, seed=42)  # 使用spring布局算法进行节点位置布局
            # nx.draw(G, pos, with_labels=True, node_size=50, font_size=10, font_color='black', node_color='skyblue',
            #         edge_color='gray')
            # plt.title('Old Graph Visualization')
            # plt.show()
            #
            # new_data = Data(x=x, edge_index=edge_index_new)
            #
            # print('edges_new_dict', edges_new_dict)
            #
            # G = to_networkx(new_data, to_undirected=True)
            #
            # # 绘制图的拓扑结构
            # plt.figure(figsize=(8, 8))
            # pos = nx.spring_layout(G, seed=42)  # 使用spring布局算法进行节点位置布局
            # nx.draw(G, pos, with_labels=True, node_size=50, font_size=10, font_color='black', node_color='skyblue',
            #         edge_color='gray')
            # plt.title('Graph Visualization')
            # plt.show()

            global_layeredge_list = [1]

            if len(changededgelist) > 0 and len(layer_edge_list) > 3:
                total+=1
                result_dict = dict()

                # print('test_layeredge_result_nonlinear',test_layeredge_result_nonlinear)
                result_dict['original KL'] = KL_original
                result_dict['len target_changed_edgelist'] = len(changededgelist)
                result_dict['len target_layer_edge_list'] = len(layer_edge_list)

                result_dict['new prob'] = softmax(output_new.detach().numpy()).tolist()
                result_dict['old prob'] = softmax(output_old.detach().numpy()).tolist()


                for idx in range(len(global_layeredge_list)):
                    select_layeredge = global_layeredge_list[idx]

                    select_layeredges_list = main_con(select_layeredge, test_layeredge_result_nonlinear,
                                                      layer_edge_list,
                                                      logit_old.detach().numpy(), output_new.detach().numpy())

                    print('select_layeredges_list', select_layeredges_list)

                    if select_layeredges_list[0][0] in in_motif_nodes_list and select_layeredges_list[0][1] in in_motif_nodes_list:
                        print('yes',target_node)
                        count+=1
                        result_dict['result']='yes'
                    else:
                        result_dict['result'] = 'false'


                    evaluate_layeredge_index1, evaluate_layeredge_index2, evaluate_layeredge_weight1, evaluate_layeredge_weight2 = from_layeredges_to_evaulate(
                        select_layeredges_list, edge_weight_old, edge_index_old, edges_old_dict, adj_old, adj_new)

                    evaluate_output = model.verify_layeredge(x, torch.tensor(evaluate_layeredge_index1), \
                                                             torch.tensor(evaluate_layeredge_index2),
                                                             edge_weight1=torch.tensor(evaluate_layeredge_weight1), \
                                                             edge_weight2=torch.tensor(
                                                                 evaluate_layeredge_weight2)).view(-1)
                    # print('evaluate_output',softmax(evaluate_output.detach().numpy()))
                    # print('output_new', softmax(output_new.detach().numpy()))

                    KL = KL_divergence(softmax(output_new.detach().numpy()),
                                       softmax(evaluate_output.detach().numpy()))

                    select_layeredges_list_str = map(lambda x: str(x), select_layeredges_list)
                    print('select layeredge KL', KL)

                    result_dict[str(idx) + ',' + 'select layeredge'] = ",".join(select_layeredges_list_str)
                    result_dict[str(idx) + ',' + 'select layeredge' + 'KL'] = KL

                os_path = f'result/house/{changed_ratio}/method'
                if not os.path.exists(os_path):
                    os.makedirs(os_path)

                json_matrix = json.dumps(result_dict)
                with open(f'result/house/{changed_ratio}/method/{target_node}.json',
                          'w') as json_file:
                    json_file.write(json_matrix)
                print('save success')

    print('count',count)
    print('total',total)
    print('explanation accuracy', count/total)
















