import torch
import matplotlib.pyplot as plt
import networkx as nx
import random
import json
import os
from torch_geometric.utils import degree
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCN
from torch_geometric.utils import k_hop_subgraph
import torch
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.datasets.motif_generator import CycleMotif
def gen_negative_graph(node_idx):
    data=dataset[node_idx]
    edges_old=data.edge_index
    #print(edges_old)
    features=data.x
    #print('features',features)
    old_labels=[]
    in_motif_nodes_list=[]
    notin_motif_nodes_list=[]
    in_motif_edges_list=[]
    notin_motif_edges_list=[]
    for i in range(features.shape[0]):
        old_labels.append(data.y[i].item())
        if data.y[i].item()==0:
            notin_motif_nodes_list.append(i)
        else:
            in_motif_nodes_list.append(i)

    print('in_motif_nodes_list',in_motif_nodes_list)

    # for i in range(features.shape[0]):
    #     edges_old[0].append(torch.tensor(i))
    #     edges_old[1].append(torch.tensor(i))

    edges_old_dict = dict()
    for i in range(len(edges_old[0])):
        if edges_old[0][i] in in_motif_nodes_list and edges_old[1][i] in in_motif_nodes_list and \
                [edges_old[0][i],edges_old[1][i]] not in in_motif_edges_list and [edges_old[1][i],edges_old[0][i]] not in in_motif_edges_list:
            in_motif_edges_list.append([edges_old[0][i].item(),edges_old[1][i].item()])
        if edges_old[0][i] not in in_motif_nodes_list or edges_old[1][i] not in in_motif_nodes_list:
            if [edges_old[0][i],edges_old[1][i]] not in notin_motif_edges_list and [edges_old[1][i],edges_old[0][i]] not in notin_motif_edges_list:
                notin_motif_edges_list.append([edges_old[0][i].item(),edges_old[1][i].item()])
        edges_old_dict[str(edges_old[0][i].item()) + ',' + str(edges_old[1][i].item())] = i
    print('edges_old_dict',edges_old_dict)
    print('in_motif_edges_list',in_motif_edges_list)
    print('notin_motif_edges_list',notin_motif_edges_list)
    print(len(edges_old_dict)/2)
    print(len(in_motif_edges_list)+len(notin_motif_edges_list))
    random.seed(42)
    number_pertubed_random = math.floor(random_changed_ratio * len(in_motif_edges_list))
    #number_pertubed_random = math.floor(motif_changed_ratio * len(notin_motif_edges_list))

    edges_old_dict_reverse = dict()
    for key, value in edges_old_dict.items():
        node_list = key.split(',')
        edges_old_dict_reverse[value] = [int(node_list[0]), int(node_list[1])]

    old_edges_weight=[1]*len(edges_old[0])


    # print(number_pertubed_motif)
    # print(number_pertubed_random)


    remove_edges_list=[]

    random_edge_list = random.sample(list(range(len(in_motif_edges_list))), 1)
    for idx in random_edge_list:
        random_edge=in_motif_edges_list[idx]
        remove_edges_list.append([random_edge[0], random_edge[1]])

    edges_new = [[], []]
    for i in range(len(edges_old[0])):
        edge_1 = edges_old[0][i].item()
        edge_2 = edges_old[1][i].item()
        if [edge_1, edge_2] not in remove_edges_list and [edge_2, edge_1] not in remove_edges_list:
            edges_new[0].append(edge_1)
            edges_new[1].append(edge_2)


    edges_new_dict = dict()
    for i in range(len(edges_new[0])):
        edges_new_dict[str(edges_new[0][i]) + ',' + str(edges_new[1][i])] = i

    print('edges_new_dict',edges_new_dict)

    edges_new_dict_reverse = dict()
    for key, value in edges_new_dict.items():
        edges_new_dict_reverse[value] = key

    new_edges_weight = [1] * len(edges_new[0])



    random_edge_list = random.sample(list(range(len(notin_motif_edges_list))), number_pertubed_random)
    for idx in random_edge_list:
        random_edge = notin_motif_edges_list[idx]
        edge_1=random_edge[0]
        edge_2=random_edge[1]
        tmp_weights = random.uniform(0.9,1)
        # print('tmp_weights',tmp_weights)
        idx1=str(edge_1)+','+str(edge_2)
        idx2 = str(edge_2) + ',' + str(edge_1)
        index1=edges_new_dict[idx1]
        index2 = edges_new_dict[idx2]
        print('index1',index1)
        print('index2', index2)

        new_edges_weight[index1] = tmp_weights
        new_edges_weight[index2] = tmp_weights


    print('remove_edges_list',remove_edges_list)
    print('len(old_edges_weight)',len(old_edges_weight))
    print('len(new_edges_weight)', len(new_edges_weight))

    # edges_new=[[],[]]
    # for i in range(len(edges_old[0])):
    #     edge_1=edges_old[0][i].item()
    #     edge_2=edges_old[1][i].item()
    #     if [edge_1,edge_2] not in remove_edges_list and [edge_2,edge_1] not in remove_edges_list:
    #         edges_new[0].append(edge_1)
    #         edges_new[1].append(edge_2)
    ######## add_edges_list

    # add_edges_list=[]
    # change_num=0
    # while change_num < number_pertubed_random:
    #     random_edge_list = random.sample(notin_motif_nodes_list, 2)
    #     # print('random_edge_list',random_edge_list)
    #     if str(random_edge_list[0])+','+str(random_edge_list[1]) not in edges_old_dict.keys()\
    #             and str(random_edge_list[1])+','+str(random_edge_list[0]) :
    #         add_edges_list.append(random_edge_list)
    #         change_num+=1
    # # print(edges_new)
    #
    # for i in range(len(add_edges_list)):
    #     # print('add_edges_list[i]',add_edges_list[i])
    #     edges_new[0].append(add_edges_list[i][0])
    #     edges_new[1].append(add_edges_list[i][1])
    #
    #     edges_new[0].append(add_edges_list[i][1])
    #     edges_new[1].append(add_edges_list[i][0])
    # # print(len(edges_new[0])/2)
    # print('add_edges_list', add_edges_list)


    edges_old_list=edges_old.tolist()

    for i in range(features.shape[0]):
        edges_new[0].append(i)
        edges_new[1].append(i)
        new_edges_weight.append(1)
        edges_old_list[0].append(i)
        edges_old_list[1].append(i)
        old_edges_weight.append(1)




    negative_data_dict = {
        'x': features.tolist(),
        'edge_index': edges_new,
        'y': data.y.tolist(),
        'edge_weight':new_edges_weight
    }

    positive_data_dict={'x': features.tolist(),
        'edge_index':edges_old_list,
        'y': data.y.tolist(),
        'edge_weight': old_edges_weight
                        }
    os_path = f'gen_house_data/{random_changed_ratio}/{node_idx}'
    if not os.path.exists(os_path):
        os.makedirs(os_path)

    json_matrix = json.dumps(negative_data_dict)
    with open(f'gen_house_data/{random_changed_ratio}/{node_idx}/negative.json',
              'w') as json_file:
        json_file.write(json_matrix)
    print('save success')

    json_matrix = json.dumps(positive_data_dict)
    with open(f'gen_house_data/{random_changed_ratio}/{node_idx}/positive.json',
              'w') as json_file:
        json_file.write(json_matrix)
    print('save success')



class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data
def initializeNodes(dataset):
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        dataset.transform = T.OneHotDegree(max_degree)

        # if max_degree < 1000:
        #     dataset.transform = T.OneHotDegree(max_degree)
        # else:
        #     deg = torch.cat(degs, dim=0).to(torch.float)
        #     mean, std = deg.mean().item(), deg.std().item()
        #     dataset.transform = NormalizedDegree(mean, std)

if __name__=='__main__':

    # dataset = ExplainerDataset(
    #     graph_generator=BAGraph(num_nodes=10, num_edges=5),
    #     motif_generator='house',
    #     num_motifs=1,
    #     transform=T.Constant(),
    #     num_graphs=1
    # )
    # data = dataset[0]
    # print(data)
    # for i in range(data.x.shape[0]):
    #     print(i,data.y[i])



    # dataset1 = ExplainerDataset(
    #     graph_generator=BAGraph(num_nodes=25, num_edges=1),
    #     motif_generator=HouseMotif(),
    #     num_motifs=1,
    #     transform=T.Constant(),
    #     num_graphs=500,
    # )
    num_graphs=1000

    dataset = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=10, num_edges=1),
        motif_generator=HouseMotif(),
        num_motifs=1,
        transform=T.Constant(),
        num_graphs=1000,
    )
    initializeNodes(dataset)
    # print(dataset[1].x)

    random_changed_ratio=0.9
    # motif_changed_ratio=0.3

    for i in range(num_graphs):
        negative_data=gen_negative_graph(i)
    # gen_negative_graph(0)



    # dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])



    G = to_networkx(dataset[0], to_undirected=True)

    # 绘制图的拓扑结构
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)  # 使用spring布局算法进行节点位置布局
    nx.draw_networkx(G, pos, with_labels=True, node_size=50, font_size=10, font_color='black', node_color='skyblue',
            edge_color='gray')
    plt.title('Graph Visualization')
    plt.show()
    #
    # G = to_networkx(negative_data, to_undirected=True)
    #
    # # 绘制图的拓扑结构
    # plt.figure(figsize=(8, 8))
    # pos = nx.spring_layout(G, seed=42)  # 使用spring布局算法进行节点位置布局
    # nx.draw(G, pos, with_labels=True, node_size=50, font_size=10, font_color='black', node_color='skyblue',
    #         edge_color='gray')
    # plt.title('Graph Visualization')
    # plt.show()

    # idx = torch.arange(data.num_nodes)
    # train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=data.y)
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data = data.to(device)
    # model = GCN(data.num_node_features, hidden_channels=20, num_layers=3,
    #             out_channels=dataset.num_classes).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
    #
    #
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