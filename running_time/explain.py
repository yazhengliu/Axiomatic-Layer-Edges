import copy
from mpi4py import MPI
from torch_geometric.datasets import Coauthor
from torch.nn.modules.module import Module
import time
import torch
from torch.nn import Parameter
import  math
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import argparse
import torch.nn as nn
import scipy.sparse as sp
from utils import rumor_construct_adj_matrix,matrixtodict,clear,softmax,KL_divergence,findnewpath,reverse_paths,test_path_contribution_layeredge,\
    sparse_mx_to_torch_sparse_tensor,main_con,map_target,from_layeredges_to_evaulate
import sys
sys.path.append('..')
from select_args import select_path_number,select_edge_number
from torch_geometric.nn import GCNConv
import os
from torch_geometric.datasets import Coauthor,Planetoid
import json

class GCN(nn.Module):
        def __init__(self, nfeat, nhid, nclass,dropout):
            super(GCN, self).__init__()

            self.gc1 = GCNConv(nfeat, nhid,normalize=False,add_self_loops=False,bias=False)
            self.gc2 = GCNConv(nhid,  nclass,normalize=False,add_self_loops=False,bias=False)
            # self.gc3 = GraphConvolution(nhid2, nclass)
            self.dropout=dropout


        def forward(self, x, adj):
            # adj=torch.tensor(adj,requires_grad=True)
            # adj = F.relu(adj)


            x = F.relu(self.gc1(x, adj))
            x=F.dropout(x,self.dropout,training=self.training)
            x = self.gc2(x, adj)
            # x = F.dropout(x, self.dropout, training=self.training)
            # x=self.gc3(x, adj)
            # print(F.softmax(x, dim=1))
            return x
        def forward_v2(self, x, adj,mask):
            # num_nodes=x.shape[0]
            # adj=torch.tensor(adj,requires_grad=True)
            # mask=self.sig(mask)
            mask = (mask + mask.t()) / 2
            adj=(adj + adj.t()) / 2
            # diag_mask=torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
            #mask=F.relu(mask)
            # print(mask.requires_grad)
            # print(adj.requires_grad)
            mask=mask*adj
            # mask=(mask + mask.t()) / 2
            x = F.relu(self.gc1(x, mask))
            x = F.dropout(x, self.dropout, training=self.training)

            x = self.gc2(x, mask)
            # x = F.dropout(x, self.dropout, training=self.training)
            # x=self.gc3(x, adj)
            # print(F.softmax(x, dim=1))
            return x
        def back(self,x,adj):
            x0 = self.gc1(x, adj)
            x1 =F.relu(x0)
            x2 = self.gc2(x1, adj)
            return (x0,x1,x2)


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
        x = self.conv2(x_1, edge_index_2, edge_weight=edgeweight2)
        return (x_0, x_1)
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def main(node_idx):
    predict_new_label = np.argmax(softmax(output_new[node_idx].detach().numpy()))

    predict_old_label = np.argmax(softmax(output_old[node_idx].detach().numpy()))

    KL_original = KL_divergence(softmax(output_new[node_idx].detach().numpy()),
                                softmax(output_old[node_idx].detach().numpy()))
    print('G_new', softmax(output_new[node_idx].detach().numpy()))
    # print('G_ceshi_new',softmax(output_ceshi_new[0].detach().numpy()))
    print('G_old', softmax(output_old[node_idx].detach().numpy()))
    # print('G_ceshi_old', softmax(output_ceshi_old[0].detach().numpy()))

    print('KL_original', KL_original)

    base_start = time.time()

    path_ceshi = findnewpath(changededgelist, graph_all, layernumbers, node_idx)
    target_path = reverse_paths(path_ceshi)

    print(len(target_path), 'target_path')

    target_changed_edgelist = []
    # print('target_path', target_path)
    for path in target_path:
        if [path[0], path[1]] in changededgelist or [path[1], path[0]] in changededgelist:
            target_changed_edgelist.append([path[0], path[1]])
        if [path[2], path[1]] in changededgelist or [path[1], path[2]] in changededgelist:
            target_changed_edgelist.append([path[1], path[2]])

    target_changed_edgelist = clear(target_changed_edgelist)

    target_layer_edge_list = []
    for edge in target_changed_edgelist:
        for layer in range(layernumbers):
            if edge[0] != edge[1]:
                target_layer_edge_list.append([edge[0], edge[1], layer])
                target_layer_edge_list.append([edge[1], edge[0], layer])
            else:
                target_layer_edge_list.append([edge[0], edge[1], layer])

    print('changededgelist', len(changededgelist))

    print('target_changed_edgelist', len(target_changed_edgelist))

    print('target_layer_edge_list', len(target_layer_edge_list))

    base_end = time.time()

    if len(target_layer_edge_list) > 50:
        result_dict = dict()
        _, _, test_layeredge_result_nonlinear, contribution_time, shapley_time = test_path_contribution_layeredge(
            target_path,
            adj_old,
            adj_new,
            target_changed_edgelist,
            relu_delta,
            relu_start,
            relu_end, x, W1, W2)
        print('base_time', base_end - base_start)
        print('contribution_time', contribution_time)
        print('shapley_time', shapley_time)

        result_dict['KL'] = KL_original
        result_dict['changed_edges'] = len(target_changed_edgelist)
        result_dict['changed_layeredges'] = len(target_layer_edge_list)
        result_dict['changes_paths'] = len(target_path)

        result_dict['base_time'] = base_end - base_start
        result_dict['contribution_time'] = contribution_time
        result_dict['shapley_time'] = shapley_time

        flag = True
        ceshi_edge_result = np.zeros((adj_new.shape[0], W2.shape[1]))
        for key, value in test_layeredge_result_nonlinear.items():
            ceshi_edge_result += value
        # print('ceshi_edge_result', ceshi_edge_result)
        true_diff_logits_nonlinear = output_new.detach().numpy() - output_old.detach().numpy()

        # print('ceshi_edge_result',ceshi_edge_result[node_idx])
        # print('true_diff_logits_nonlinear',true_diff_logits_nonlinear[node_idx])

        if true_diff_logits_nonlinear[node_idx].any() != 0:
            if np.all(abs(ceshi_edge_result[node_idx] - true_diff_logits_nonlinear[node_idx]) > 1e-4):
                print('key', node_idx, 'test', ceshi_edge_result[node_idx], 'true',
                      true_diff_logits_nonlinear[node_idx])
                flag = False

        print('layeredge flag', flag)

        target_layeredge_result = map_target(test_layeredge_result_nonlinear, node_idx)

        global_layeredge_list = select_path_number(modelname,  len(target_layer_edge_list))

        for idx in range(len(global_layeredge_list)):
            select_start_time = time.time()
            select_layeredge = global_layeredge_list[idx]

            print('i', idx, 'select_layeredge', select_layeredge)

            # print(output_new.shape)
            select_layeredges_list = main_con(select_layeredge, node_idx, target_layeredge_result,
                                              target_layer_edge_list,
                                              output_old[node_idx].detach().numpy(), output_new,num_class)
            select_end_time = time.time()
            result_dict[f'select_{idx}'] = select_end_time - select_start_time

            evaluate_layeredge_index1, evaluate_layeredge_index2, evaluate_layeredge_weight1, evaluate_layeredge_weight2 = from_layeredges_to_evaulate(
                select_layeredges_list, edges_weight_old, edges_old, edges_old_dict, adj_old, adj_new)

            # print('weight1',evaluate_layeredge_weight1[ edges_old_dict[str(7) + ',' + str(7)]])
            # print('weight2',evaluate_layeredge_weight2[edges_old_dict[str(7) + ',' + str(7)]])

            evaluate_output = model_gnn.forward(x, torch.tensor(evaluate_layeredge_index1), \
                                                torch.tensor(evaluate_layeredge_index2),
                                                edge_weight1=torch.tensor(evaluate_layeredge_weight1), \
                                                edge_weight2=torch.tensor(evaluate_layeredge_weight2))

            KL = KL_divergence(softmax(output_new[node_idx].detach().numpy()),
                               softmax(evaluate_output[node_idx].detach().numpy()))
            select_layeredges_list_str = map(lambda x: str(x), select_layeredges_list)
            print('select layeredge KL', KL)

            result_dict[str(idx) + ',' + 'select layeredge'] = ",".join(select_layeredges_list_str)
            result_dict[str(idx) + ',' + 'select layeredge' + 'KL'] = KL

        print('result_dict', result_dict)

        os_path = f'../result/running_time/{modelname}/{changed_ratio}'
        if not os.path.exists(os_path):
            os.makedirs(os_path)

        json_matrix = json.dumps(result_dict)
        with open(f'../result/running_time/{modelname}/{changed_ratio}/{node_idx}.json',
                  'w') as json_file:
            json_file.write(json_matrix)
        print('save success')

def bigjobMPI(node_list):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    #print('size',size)
    print('rank',rank)



    numjobs = len(node_list)
    #print('numjobs',numjobs)


    # arrange the works and jobs
    if rank == 0:
        # this is head worker
        # jobs are arranged by this worker
        job_all_idx = list(range(numjobs))
        # print('job_all_idx', job_all_idx)
        # random.shuffle(job_all_idx)

        # shuffle the job index to make all workers equal
        # for unbalanced jobs
    else:
        job_all_idx = None

    job_all_idx = comm.bcast(job_all_idx, root=0)

    njob_per_worker = int(numjobs / size)
    # the number of jobs should be a multiple of the NumProcess[MPI]

    this_worker_job = [job_all_idx[x] for x in range(rank * njob_per_worker, (rank + 1) * njob_per_worker)]
    print('this_worker_job',this_worker_job)

    # map the index to parameterset [eps,anis]
    work_content = [node_list[x] for x in this_worker_job]
    print('work_content[0]',work_content[0])

    for a_piece_of_work in work_content:
        print('a_piece_of_work',a_piece_of_work)
        main(a_piece_of_work)

if __name__ == "__main__":
    dataset = Coauthor('../data/Physics', 'Physics')
    modelname = 'physics'

    # dataset = Coauthor('../data/Cs', 'Cs')
    # modelname = 'cs'

    # dataset = Planetoid('../data/pubmed', 'PubMed')
    # modelname = 'pubmed'

    layernumbers=2

    data = dataset[0]
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden1', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--hidden2', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate (1-keep probability)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # print(data)
    x, edge_index = data.x, data.edge_index

    # print('numclasses')

    print(x.shape[0])
    #
    #
    labels = data.y

    num_class=labels.max().item() + 1
    print('num_class',num_class)
    model = GCN(nfeat=x.shape[1],
                nhid=args.hidden1,
                nclass=labels.max().item() + 1,
                dropout=args.dropout
                )
    model.eval()
    model.load_state_dict(torch.load(f'../data/{modelname}/{modelname}_GCNmodel.pkl'))

    print(model.state_dict().keys())

    model_gnn = GCN_test(nfeat=x.shape[1],
                             nhid=args.hidden1,
                             nclass=labels.max().item() + 1,
                             dropout=args.dropout)
    model_gnn.eval()

    model_dict = model_gnn.state_dict()
    #print(model.state_dict())
    model_dict['conv1.lin.weight'] = model.state_dict()['gc1.lin.weight']
    model_dict['conv2.lin.weight'] = model.state_dict()['gc2.lin.weight']
    model_gnn.load_state_dict(model_dict)



    edges_old=copy.deepcopy(edge_index.tolist())
    for i in range(x.shape[0]):
        edges_old[0].append(i)
        edges_old[1].append(i)


    #print(edges_old)

    adj_old = rumor_construct_adj_matrix(edges_old,x.shape[0])
    # print(adj_old)
    adj_old_nonzero=adj_old.nonzero()

    if (adj_old.todense() == adj_old.todense().T).all():
        print("adj_start是对称矩阵。")
    else:
        print("adj_start不是对称矩阵。")

    # adj_old_todense=adj_old.todense()
    # print(adj_old_todense.shape)
    # for i in range(labels.shape[0]):
    #     for j in range(i+1,labels.shape[0]):
    #         if adj_old_todense[i,j]!=adj_old_todense[j,i]:
    #             print('adj_old_todense',adj_old_todense[i,j],i,j,adj_old_todense[j,i])

    # if (adj_end == adj_end.T).all():
    #     print("adj_end是对称矩阵。")
    # else:
    #     print("adj_end不是对称矩阵。")


    # train_ratio = 0.5
    # val_ratio = 0.3
    #
    # idx_train, idx_val, idx_test = split_train_test(labels,train_ratio,val_ratio)
    #
    # adj= construct_adj_matrix(edge_index, labels)
    # adj = normalize(adj + sp.eye(len(labels)))
    # print(adj)
    # adj_old=adj.todense()
    #
    # adj_old_nonzero = adj.nonzero()
    # graph = matrixtodict(adj_old_nonzero)
    # print(graph)
    #
    # adj_sp = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # print(accuracy(model.forward(x,adj_sp)[idx_test],labels[idx_test]))

    edges_weight_old = []
    edges_old_dict = dict()

    for i in range(len(edges_old[0])):
        edges_weight_old.append(adj_old[edges_old[0][i], edges_old[1][i]])
        edges_old_dict[str(edges_old[0][i]) + ',' + str(edges_old[1][i])] = i
    edges_weight_old = torch.tensor(edges_weight_old)


    edges_old_dict_reverse = dict()
    for key, value in edges_old_dict.items():
        node_list = key.split(',')
        edges_old_dict_reverse[value] = [int(node_list[0]), int(node_list[1])]
    #print('edges_old_dict_reverse',edges_old_dict_reverse)


    changed_ratio=0.2

    number_pertubed=math.floor(changed_ratio*len(edges_old_dict)/2)

    print('number_pertubed',number_pertubed)

    change_num = 0
    changededgelist = []

    adj_new = copy.deepcopy(adj_old)
    random.seed(42)

    random_edge_list = random.sample(list(range(len(edges_old_dict))), number_pertubed)
    # print('random_edge_list ',random_edge_list,change_num)
    for i in range(len(random_edge_list)):
        if i%1000==0:
            print('i',i)
        random_edge=random_edge_list[i]
        # print('random value',(b - a) * np.random.random() + a)
        remove_node_list = edges_old_dict_reverse[random_edge]

        tmp_weights = random.random()
        if adj_old[remove_node_list[0], remove_node_list[1]] != tmp_weights and [
            remove_node_list[0], remove_node_list[1]] not in changededgelist \
                and [remove_node_list[1], remove_node_list[0]] not in changededgelist:
            change_weight = tmp_weights
            adj_new[remove_node_list[0], remove_node_list[1]] = change_weight
            adj_new[remove_node_list[1], remove_node_list[0]] = change_weight

            changededgelist.append([remove_node_list[0], remove_node_list[1]])
            change_num += 1




    print('len(changededgelist)', len(changededgelist))

    # changededgelist = clear(changededgelist)
    # # print('changededgelist', changededgelist)
    # print('len(changededgelist)',len(changededgelist))

    adj_new_nonzero = adj_new.nonzero()

    edges_new = copy.deepcopy(edges_old)

    edges_new_dict = dict()
    edges_weight_new = []
    for i in range(len(edges_new[0])):
        edges_weight_new.append(adj_new[edges_new[0][i], edges_new[1][i]])
        edges_new_dict[str(edges_new[0][i]) + ',' + str(edges_new[1][i])] = i
    edges_weight_new = torch.tensor(edges_weight_new)


    print('edges_new_dict',len(edges_new_dict))
    print('edges_old_dict', len(edges_old_dict))



    # print(len(edges_old[0]))

    print(len(adj_old_nonzero[0]))
    print(len(adj_new_nonzero[0]))




    graph_old = matrixtodict(adj_old_nonzero)
    graph_new = matrixtodict(adj_new_nonzero)

    graph_all = copy.deepcopy(graph_old)
    # print('graph_all', graph_all)
    for edge in changededgelist:
        if edge[1] not in graph_all[edge[0]]:
            graph_all[edge[0]].append(edge[1])
        if edge[0] not in graph_all[edge[1]]:
            graph_all[edge[1]].append(edge[0])

    #print('graph_all',graph_all)

    edges_old=torch.tensor(edges_old)
    edges_new=torch.tensor(edges_new)

    output_old=model_gnn.forward(x,edges_old,edges_old,edges_weight_old,edges_weight_old)
    output_new=model_gnn.forward(x,edges_new,edges_new,edges_weight_new,edges_weight_new)

    W1 = model_gnn.state_dict()['conv1.lin.weight'].t()
    W2 = model_gnn.state_dict()['conv2.lin.weight'].t()

    target_node_list=[]
    for i in range(x.shape[0]):
        KL_original=KL_divergence(softmax(output_new[i].detach().numpy()),softmax(output_old[i].detach().numpy()))
        if KL_original>0.2:
            target_node_list.append(i)

    print('target_node_list',len(target_node_list),target_node_list)

    nonlinear_start_layer1, nonlinear_relu_start_layer1 = model_gnn.back(x, edges_old, edges_old,
                                                                         edges_weight_old, edges_weight_old)
    nonlinear_end_layer1, nonlinear_relu_end_layer1 = model_gnn.back(x, edges_new, edges_new,
                                                                     edges_weight_new, edges_weight_new)

    # print('nonlinear_start_layer1',nonlinear_start_layer1)

    relu_delta = torch.where((nonlinear_end_layer1 - nonlinear_start_layer1) != 0,
                             (nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1) / (
                                     nonlinear_end_layer1 - nonlinear_start_layer1),
                             torch.zeros_like((nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1)))
    relu_end = torch.where((nonlinear_end_layer1) != 0, nonlinear_relu_end_layer1 / nonlinear_end_layer1,
                           torch.zeros_like(nonlinear_end_layer1))
    relu_start = torch.where((nonlinear_start_layer1) != 0, nonlinear_relu_start_layer1 / nonlinear_start_layer1,
                             torch.zeros_like(nonlinear_start_layer1))

    layer_edge_list = []
    for edge in changededgelist:
        for layer in range(layernumbers):
            if edge[0] != edge[1]:
                layer_edge_list.append([edge[0], edge[1], layer])
                layer_edge_list.append([edge[1], edge[0], layer])
            else:
                layer_edge_list.append([edge[0], edge[1], layer])

    print(len(target_node_list))

    main(0)

    # bigjobMPI(range(x.shape[0]))
    # pass





























