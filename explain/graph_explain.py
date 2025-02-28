from torch_geometric.datasets import TUDataset
import torch
import math
import copy
from  dig.xgraph.dataset import MoleculeDataset
from .explain_utils import split,rumor_construct_adj_matrix,difference_weight,clear,matrixtodict,softmax,KL_divergence,findnewpath,\
    reverse_paths,contribution_layeredge,contribution_edge,find_target_changed_layer_edegs,solve_layeredge_graph,from_layeredges_to_evaulate,from_edges_to_evaulate,\
    solve_edge_graph
import os,json
import numpy as np

import sys
sys.path.append('..')
from select_args import select_path_number,select_edge_number
from train.train_graph import GCN,initializeNodes
class gen_graph_data():
    def __init__(self,dataset,index,args):
        self.dataset=dataset
        self.index=index
        self.args=args
    def gen_original_edge(self):
        data = self.dataset[self.index]
        x, edge_index = data.x, data.edge_index
        edge_index_list = data.edge_index.numpy().tolist()
        # for i in range(0, x.shape[0]):
        #     edge_index_list[0].append(i)
        #     edge_index_list[1].append(i)

        edge_index = torch.tensor(edge_index_list)
        return data,x,edge_index

    def pertub_edges(self,data,edge_index_all):
        edge_time_result = dict()
        for i in range(len(edge_index_all[0])):
            node1 = edge_index_all[0][i].item()
            node2 = edge_index_all[1][i].item()
            if str(node1) + ',' + str(node2) not in edge_time_result.keys() and str(node2) + ',' + str(
                        node1) not in edge_time_result.keys():
                    edge_time_result[str(node1) + ',' + str(node2)] = 1


        model_name=self.args.dataset
        sort_edge_time_result = sorted(edge_time_result.items(), key=lambda x: x[1])
        if model_name=='mutag':
            sliding_T = math.floor(len(sort_edge_time_result) / 4)
            sort_edge_time_result = sorted(edge_time_result.items(), key=lambda x: x[1])
            edge_index_old = split(0, sliding_T * 3, sort_edge_time_result,
                                   torch.tensor(range(data.num_nodes)))
            edge_index_new = split(sliding_T, len(sort_edge_time_result), sort_edge_time_result,
                                   torch.tensor(range(data.num_nodes)))
        elif model_name=='clintox':
            sliding_T = math.floor(len(sort_edge_time_result) / 5)

            sort_edge_time_result = sorted(edge_time_result.items(), key=lambda x: x[1])

            edge_index_old = split(0, sliding_T * 2, sort_edge_time_result,
                                   torch.tensor(range(data.num_nodes)))
            # print(edge_index_old)

            edge_index_new = split(sliding_T, len(sort_edge_time_result), sort_edge_time_result,
                                   torch.tensor(range(data.num_nodes)))
        elif model_name=='IMDB-BINARY':
            sliding_T = math.floor(len(sort_edge_time_result) / 3)

            sort_edge_time_result = sorted(edge_time_result.items(), key=lambda x: x[1])

            edge_index_old = split(0, sliding_T * 2, sort_edge_time_result,
                                   torch.tensor(range(data.num_nodes)))
            # print(edge_index_old)

            edge_index_new = split(sliding_T, len(sort_edge_time_result), sort_edge_time_result,
                                   torch.tensor(range(data.num_nodes)))
        elif model_name=='REDDIT-BINARY':
            sliding_T = math.floor(len(sort_edge_time_result) / 3)

            sort_edge_time_result = sorted(edge_time_result.items(), key=lambda x: x[1])

            edge_index_old = split(0, sliding_T * 2, sort_edge_time_result,
                                   torch.tensor(range(data.num_nodes)))
            # print(edge_index_old)

            edge_index_new = split(sliding_T, len(sort_edge_time_result), sort_edge_time_result,
                                   torch.tensor(range(data.num_nodes)))

        adj_old = rumor_construct_adj_matrix(edge_index_old, data.num_nodes)
        adj_new = rumor_construct_adj_matrix(edge_index_new, data.num_nodes)

        adj_new_nonzero = adj_old.nonzero()
        adj_old_nonzero = adj_new.nonzero()

        edge_weight_old = []
        edges_old_dict = dict()

        for i in range(len(edge_index_old[0])):
            edge_weight_old.append(adj_old[edge_index_old[0][i], edge_index_old[1][i]])
            edges_old_dict[str(edge_index_old[0][i]) + ',' + str(edge_index_old[1][i])] = i
        edge_weight_old = torch.tensor(edge_weight_old)

        edge_weight_new = []
        edges_new_dict = dict()

        for i in range(len(edge_index_new[0])):
            edge_weight_new.append(adj_new[edge_index_new[0][i], edge_index_new[1][i]])
            edges_new_dict[str(edge_index_new[0][i]) + ',' + str(edge_index_new[1][i])] = i
        edge_weight_new = torch.tensor(edge_weight_new)

        changededgelist = difference_weight(edge_index_new, edge_index_old, adj_new, adj_old)
        # print(addedgelist)
        changededgelist = clear(changededgelist)

        graph_old = matrixtodict(adj_old_nonzero)
        graph_new = matrixtodict(adj_new_nonzero)

        # print('graph_old',graph_old)

        graph_all = copy.deepcopy(graph_old)
        # print('graph_all', graph_all)
        for edge in changededgelist:
            if edge[1] not in graph_all[edge[0]]:
                graph_all[edge[0]].append(edge[1])
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

        return adj_old, adj_new, edge_index_old, edge_index_new, graph_old, graph_new, graph_all, changededgelist, edge_weight_old,edge_weight_new, edges_old_dict, edges_new_dict, edges_old_dict_reverse, edges_new_dict_reverse


    def gen_model(self,model_path,dataset):
        model=GCN(nfeat=dataset.num_features,hidden_channels=16,nclass=2)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def gen_parameters(self,model,x_tensor,edges_new,edges_old,edgeweight1,edgeweight2):
        model.eval()

        nonlinear_start_layer1, nonlinear_relu_start_layer1 = model.back(x_tensor, edges_old,
                                                                         edgeweight1,)
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

def explain_graph(args):
    model_name =args.dataset
    if model_name=='mutag':
        dataset = TUDataset('data', name='MUTAG', use_node_attr='True')
    elif model_name=='clintox':
        dataset = MoleculeDataset('data', model_name)
    elif model_name=='IMDB-BINARY' or model_name=='REDDIT-BINARY':
        dataset = TUDataset('data', name=args.dataset, use_node_attr='True')
        initializeNodes(dataset)


    graph_index_list = list(range(len(dataset)))

    for target_index in graph_index_list:
        old_data = gen_graph_data(dataset, target_index,args)
        data, x, edge_index_all = old_data.gen_original_edge()
        x = x.to(torch.float32)
        print('data.num_nodes', data.num_nodes)

        adj_old, adj_new, edge_index_old, edge_index_new, graph_old, graph_new, graph_all, changededgelist, edges_weight_old, edges_weight_new, edges_old_dict, edges_new_dict, edges_old_dict_reverse, edges_new_dict_reverse = old_data.pertub_edges(
            data, edge_index_all)

        model_path = f'data/{model_name}/GCN_model.pth'

        model = old_data.gen_model(model_path,dataset)
        model.eval()

        W1, W2, relu_delta, relu_end, relu_start = old_data.gen_parameters(model, x, edge_index_new, edge_index_old,
                                                                           edges_weight_old, edges_weight_new)

        output_old = model.forward(x, edge_index_old, edges_weight_old).view(-1)
        output_new = model.forward(x, edge_index_new, edges_weight_new).view(-1)

        logit_old = model.pre_forward(x, edge_index_old, edges_weight_old)

        logit_new = model.pre_forward(x, edge_index_new, edges_weight_new)

        KL_original = KL_divergence(softmax(output_new.detach().numpy()),
                                    softmax(output_old.detach().numpy()))

        all_changed_paths=[]
        for node in range(x.shape[0]):
            all_changed_paths = all_changed_paths + findnewpath(changededgelist, graph_all, args.layernumbers, node)
        target_path = reverse_paths(all_changed_paths)

        target_changed_layeredges=find_target_changed_layer_edegs(changededgelist,args.layernumbers)



        _, _, layeredge_result = contribution_layeredge(target_path,
                                                                                 adj_old,
                                                                                 adj_new,
                                                                                 changededgelist,
                                                                                 relu_delta,
                                                                                 relu_start,
                                                                                 relu_end, x, W1, W2)

        summation_to_delta_layeredge_flag = True
        ceshi_edge_result = np.zeros((adj_old.shape[0], 2))
        for key, value in layeredge_result.items():
            ceshi_edge_result += value

        true_diff_logits_nonlinear = logit_new.detach().numpy() - logit_old.detach().numpy()
        # print('ceshi_edge_result', ceshi_edge_result)
        # print('diff',true_diff_logits_nonlinear)
        for i in range(adj_old.shape[0]):
            if true_diff_logits_nonlinear[i].any() != 0:
                if np.any(abs(ceshi_edge_result[i] - true_diff_logits_nonlinear[i]) > 1e-4):
                    print('key', i, 'test', ceshi_edge_result[i], 'true', true_diff_logits_nonlinear[i])
                    summation_to_delta_layeredge_flag = False
        print('layeredge flag', summation_to_delta_layeredge_flag )

        _, _, edge_result = contribution_edge(target_path,
                                                             adj_old,
                                                             adj_new,
                                                             changededgelist,
                                                             relu_delta,
                                                             relu_start,
                                                             relu_end, x, W1, W2)

        summation_to_delta_edge_flag=True

        ceshi_edge_result = np.zeros((adj_old.shape[0], 2))
        for key, value in edge_result.items():
            ceshi_edge_result += value
        true_diff_logits_nonlinear = logit_new.detach().numpy() - logit_old.detach().numpy()
        for i in range(adj_old.shape[0]):
            if true_diff_logits_nonlinear[i].any() != 0:
                if np.all(abs(ceshi_edge_result[i] - true_diff_logits_nonlinear[i]) > 1e-4):
                    print('key', i, 'test', ceshi_edge_result[i], 'true', true_diff_logits_nonlinear[i])
                    summation_to_delta_edge_flag = False

        print('edge flag', summation_to_delta_edge_flag)

        global_layeredge_list = select_path_number(args.dataset, len(layeredge_result))

        global_edge_list = select_edge_number(args.dataset, len(changededgelist))

        print('global_edge_list',global_edge_list)

        if len(changededgelist) > 5 and len(target_changed_layeredges) > 20 :
            result_dict = dict()
            result_dict['original KL'] = KL_original
            result_dict['len target_changed_edgelist'] = len(changededgelist)
            result_dict['len target_layer_edge_list'] = len(target_changed_layeredges)
            result_dict['new prob'] = softmax(output_new.detach().numpy()).tolist()
            result_dict['old prob'] = softmax(output_old.detach().numpy()).tolist()
            for idx in range(len(global_layeredge_list)):
                select_layeredge = global_layeredge_list[idx]

                select_layeredges_list = solve_layeredge_graph(select_layeredge, layeredge_result,
                                                  target_changed_layeredges,
                                                  logit_old.detach().numpy(), output_new.detach().numpy())

                evaluate_layeredge_index1, evaluate_layeredge_index2, evaluate_layeredge_weight1, evaluate_layeredge_weight2 = from_layeredges_to_evaulate(
                    select_layeredges_list, edges_weight_old, edge_index_old.tolist(), edges_old_dict, adj_old, adj_new)

                evaluate_output = model.verify_layeredge(x, torch.tensor(evaluate_layeredge_index1), \
                                                         torch.tensor(evaluate_layeredge_index2),
                                                         edge_weight1=torch.tensor(evaluate_layeredge_weight1), \
                                                         edge_weight2=torch.tensor(evaluate_layeredge_weight2)).view(-1)
                # print('evaluate_output',softmax(evaluate_output.detach().numpy()))
                # print('output_new', softmax(output_new.detach().numpy()))

                KL = KL_divergence(softmax(output_new.detach().numpy()),
                                   softmax(evaluate_output.detach().numpy()))

                select_layeredges_list_str = map(lambda x: str(x), select_layeredges_list)
                print('select layeredge KL', KL)

                result_dict[str(idx) + ',' + 'select layeredge'] = ",".join(select_layeredges_list_str)
                result_dict[str(idx) + ',' + 'select layeredge' + 'KL'] = KL

            for idx in range(len(global_edge_list)):
                select_edge = global_edge_list[idx]
                print('i', idx, 'select_edge', select_edge)
                select_edges_list = solve_edge_graph(select_edge, edge_result, changededgelist,
                                                  logit_old.detach().numpy(), output_new.detach().numpy())

                print('select_edges_list ', select_edges_list)
                evaluate_edge_index, evaluate_edge_weight = from_edges_to_evaulate(select_edges_list, edges_weight_old,
                                                                                   edge_index_old.tolist(), edges_old_dict,
                                                                                   adj_old, adj_new)

                evaluate_edge_output = model.verify_layeredge(x, torch.tensor(evaluate_edge_index), \
                                                              torch.tensor(evaluate_edge_index),
                                                              edge_weight1=torch.tensor(evaluate_edge_weight), \
                                                              edge_weight2=torch.tensor(evaluate_edge_weight)).view(-1)

                KL_edge = KL_divergence(softmax(output_new.detach().numpy()),
                                        softmax(evaluate_edge_output.detach().numpy()))

                print('select edge KL', KL_edge)

                select_edges_list_str = map(lambda x: str(x), select_edges_list)
                result_dict[str(idx) + ',' + 'select edge'] = ",".join(select_edges_list_str)
                result_dict[str(idx) + ',' + 'select edge' + 'KL'] = KL_edge

            os_path = f'result/{args.dataset}'
            if not os.path.exists(os_path):
                os.makedirs(os_path)

            json_matrix = json.dumps(result_dict)
            with open(f'result/{args.dataset}/{target_index}.json',
                      'w') as json_file:
                json_file.write(json_matrix)
            print('save success')



