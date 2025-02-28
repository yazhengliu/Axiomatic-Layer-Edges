import json
import random

import torch
import numpy as np
import argparse
import scipy.sparse as sp
import os
import copy
import cvxpy as cvx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from  .explain_utils import rumor_construct_adj_matrix,matrixtodict,difference_weight,clear,GCN_test,from_edge_findpaths,\
find_target_changed_paths,find_target_changed_edges,find_target_changed_layer_edegs,contribution_layeredge,\
contribution_edge,map_target,KL_divergence,softmax,solve_layeredge,solve_edge,from_edges_to_evaulate,from_layeredges_to_evaulate

import sys
sys.path.append('..')
from select_args import select_path_number,select_edge_number
from train.train_pheme import Net_rumor

class gen_rumor_data():
    def __init__(self, dataset,data_path,embedding_path,model_path):
        self.dataset=dataset
        self.data_path=data_path
        self.embedding_path=embedding_path
        self.model_path=model_path
    def gen_edge_index_old(self,file_index):

        _, file_map, file_map_reverse = self.gen_idxlist()
        file_name = file_map_reverse[file_index]
        jsonPath = f'data/{self.dataset}/{self.dataset}_json/{file_name}.json'
        with open(jsonPath, 'r') as f:
            data = json.load(f)

        # x = np.array(data['x'])
        edges_old = data['edges_2']  # 0,2/3  contains 2/3 edges from the start time.
        # print('edges_old',edges_old)
        adj_old = rumor_construct_adj_matrix(edges_old, len(data['node_map']))
        # print('adj_old',adj_old)
        adj_old_nonzero = adj_old.nonzero()

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

        graph_old = matrixtodict(adj_old_nonzero)

        return adj_old,  edges_old,  graph_old,  edges_weight_old,  edges_old_dict

    def gen_edge_index_new(self,file_index):
        _, file_map, file_map_reverse = self.gen_idxlist()
        file_name = file_map_reverse[file_index]
        jsonPath = f'data/{self.dataset}/{self.dataset}_json/{file_name}.json'
        with open(jsonPath, 'r') as f:
            data = json.load(f)

        edges_new = data['edges_4']

        adj_new = rumor_construct_adj_matrix(edges_new, len(data['node_map']))
        # print('adj_old',adj_old)
        adj_new_nonzero = adj_new.nonzero()

        edges_weight_new = []
        edges_new_dict = dict()

        for i in range(len(edges_new[0])):
            edges_weight_new.append(adj_new[edges_new[0][i], edges_new[1][i]])
            edges_new_dict[str(edges_new[0][i]) + ',' + str(edges_new[1][i])] = i
        edges_weight_new = torch.tensor(edges_weight_new)
        graph_new = matrixtodict(adj_new_nonzero)

        return  adj_new, edges_new, graph_new,   edges_weight_new, edges_new_dict

    def find_changed_edges(self,edges_old,edges_new,adj_new,adj_old,graph_old):
        changededgelist = difference_weight(edges_new, edges_old, adj_new, adj_old)
        # print(addedgelist)
        changededgelist = clear(changededgelist)
        graph_all = copy.deepcopy(graph_old)
        # print('graph_all', graph_all)
        for edge in changededgelist:
            if edge[1] not in graph_all[edge[0]]:
                graph_all[edge[0]].append(edge[1])
            if edge[0] not in graph_all[edge[1]]:
                graph_all[edge[1]].append(edge[0])

        return changededgelist,graph_all

    def gen_idxlist(self):
        files_name = [file.split('.')[0] for file in os.listdir(self.data_path)]
        files_name=sorted(files_name)
        file_map = dict()
        for i in range(0, len(files_name)):
            file_map[files_name[i]] = i
        # print(file_map)
        idx_list = list(range(max(file_map.values()) + 1))
        file_map_reverse = {value: key for key, value in file_map.items()}
        return idx_list,file_map,file_map_reverse


    def gen_parameters(self,file_index,model,edges_new,edges_old,file_map_reverse,edgeweight1,edgeweight2):
        # _, file_map, file_map_reverse = self.gen_idxlist()
        file_name = file_map_reverse[file_index]
        jsonPath = f'data/{self.dataset}/{self.dataset}_json/{file_name}.json'
        with open(jsonPath, 'r') as f:
            data = json.load(f)

        edges_new_tensor = torch.tensor(edges_new)
        edges_old_tensor = torch.tensor(edges_old)


        sentence = np.array(data['intput sentenxe'])
        sentence = torch.LongTensor(sentence)
        # print('sentence',sentence)
        model.eval()
        x_tensor = model.feature(sentence)
        # x = x_tensor.detach().numpy()

        nonlinear_start_layer1, nonlinear_relu_start_layer1 = model.back(x_tensor, edges_old_tensor,edges_old_tensor,edgeweight1,edgeweight1)
        nonlinear_end_layer1, nonlinear_relu_end_layer1 =  model.back(x_tensor, edges_new_tensor,edges_new_tensor,edgeweight2,edgeweight2)

        # print('nonlinear_start_layer1',nonlinear_start_layer1)

        relu_delta = torch.where((nonlinear_end_layer1 - nonlinear_start_layer1 )!= 0, (nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1) / (nonlinear_end_layer1 - nonlinear_start_layer1 ), torch.zeros_like((nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1)))
        relu_end=torch.where((nonlinear_end_layer1)!=0,nonlinear_relu_end_layer1/nonlinear_end_layer1, torch.zeros_like(nonlinear_end_layer1))
        relu_start = torch.where((nonlinear_start_layer1) != 0, nonlinear_relu_start_layer1 / nonlinear_start_layer1,
                               torch.zeros_like(nonlinear_start_layer1))

        model.eval()
        W1 = model.state_dict()['conv1.lin.weight'].t()
        W2 = model.state_dict()['conv2.lin.weight'].t()


        return x_tensor,W1,W2 ,relu_delta,relu_end,relu_start
    def gen_embedding(self):
        embedding_numpy = np.load(self.embedding_path, allow_pickle=True)
        print('embedding_numpy success')
        embedding_tensor = torch.FloatTensor(embedding_numpy)
        return embedding_tensor
    def gen_model(self,embedding_tensor):


        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--epochs', type=int, default=200,
                            help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=16,
                            help='Number of hidden units.')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--glove_embedding', type=float, default=embedding_tensor,
                            )
        parser.add_argument('--num_layers', type=int, default=2,
                            )

        args = parser.parse_args()

        model = Net_rumor(
            nhid=args.hidden,
            nclass=2,
            dropout=args.dropout,args=args)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        # # print(model.state_dict())
        #
        # model_gnn = GCN_test(nfeat=args.hidden * 2,
        #                      nhid=args.hidden,
        #                      nclass=2,
        #                      dropout=args.dropout,edge_weight1=edgeweight1,edge_weight2=edgeweight2)
        # model_gnn.eval()
        #
        # model_dict = model_gnn.state_dict()
        # #print(model_dict)
        # model_dict['gc1.weight'] = model.state_dict()['conv1.lin.weight'].t()
        # model_dict['gc2.weight'] = model.state_dict()['conv2.lin.weight'].t()
        # model_gnn.load_state_dict(model_dict)
        return model

    @classmethod
    def gen_evaluate_model(self,model):
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--epochs', type=int, default=200,
                            help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=16,
                            help='Number of hidden units.')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate (1 - keep probability).')

        parser.add_argument('--num_layers', type=int, default=2,
                            )

        args = parser.parse_args()
        model_gnn = GCN_test(nfeat=args.hidden * 2,
                             nhid=args.hidden,
                             nclass=2,
                             dropout=args.dropout)
        model_gnn.eval()

        model_dict = model_gnn.state_dict()
        # print(model_dict)
        model_dict['conv1.lin.weight'] = model.state_dict()['conv1.lin.weight']
        model_dict['conv2.lin.weight'] = model.state_dict()['conv2.lin.weight']
        model_gnn.load_state_dict(model_dict)
        return model_gnn

def explain_rumor(args):
    data_path = f'data/{args.dataset}/{args.dataset}_json'
    embedding_path = f'data/{args.dataset}/{args.dataset}_embedding.npy'
    model_path = f'data/{args.dataset}/{args.dataset}_GCN_model.pth'
    rumor_data = gen_rumor_data(args.dataset, data_path, embedding_path, model_path)

    idx_list, _, file_map_reverse = rumor_data.gen_idxlist()
    clear_goallist = idx_list

    embedding_tensor = rumor_data.gen_embedding()
    model = rumor_data.gen_model(embedding_tensor)
    evaluate_model = rumor_data.gen_evaluate_model(model=model,
                                                   )

    target_file_list = random.sample(clear_goallist, 100)

    # print('target_file_list',target_file_list)


    for index in target_file_list:
        adj_old, edges_old, graph_old, edges_weight_old, edges_old_dict = rumor_data.gen_edge_index_old(
            clear_goallist[index])

        adj_new, edges_new, graph_new, edges_weight_new, edges_new_dict = rumor_data.gen_edge_index_new(
            clear_goallist[index])

        changededgelist, graph_all = rumor_data.find_changed_edges(edges_old, edges_new, adj_new, adj_old,
                                                                   graph_old)

        x_tensor, W1, W2, relu_delta, relu_end, relu_start = rumor_data.gen_parameters(clear_goallist[index], model,
                                                                                       edges_new,
                                                                                       edges_old, file_map_reverse,
                                                                                       edges_weight_old,
                                                                                       edges_weight_new)
        changed_paths = from_edge_findpaths(changededgelist, graph_all)

        # print('changed_paths', changed_paths)
        target_changhed_path = find_target_changed_paths(changed_paths,
                                                                       0)  # node 0 is the target node to explain in each files, denote the text
        target_changed_edgelist = find_target_changed_edges(target_changhed_path, changededgelist)
        target_changed_layeredges = find_target_changed_layer_edegs(target_changed_edgelist,
                                                                                  args.layernumbers)

        model.eval()

        output_new = model.forward_v2(x_tensor, torch.tensor(edges_new), torch.tensor(edges_new), edges_weight_new,
                                      edges_weight_new)
        output_old = model.forward_v2(x_tensor, torch.tensor(edges_old), torch.tensor(edges_old), edges_weight_old,
                                      edges_weight_old)

        _, _, layeredge_result = contribution_layeredge(target_changhed_path,
                                                                      adj_old,
                                                                      adj_new,
                                                                      target_changed_edgelist,
                                                                      relu_delta,
                                                                      relu_start,
                                                                      relu_end, x_tensor, W1, W2)

        summation_to_delta_layeredge_flag = True
        ceshi_edge_result = np.zeros((adj_new.shape[0], W2.shape[1]))
        for key, value in layeredge_result.items():
            ceshi_edge_result += value
        # print('ceshi_edge_result', ceshi_edge_result)
        true_diff_logits = output_new.detach().numpy() - output_old.detach().numpy()
        # print('true_diff_logits_nonlinear',true_diff_logits)
        for i in range(1):
            if true_diff_logits[i].any() != 0:
                if np.any(abs(ceshi_edge_result[i] - true_diff_logits[i]) > 1e-4):
                    print('key', i, 'test', ceshi_edge_result[i], 'true', true_diff_logits[i])
                    summation_to_delta_layeredge_flag = False
        print('layeredge flag', summation_to_delta_layeredge_flag)

        _, _, edge_result = contribution_edge(target_changhed_path,
                                                            adj_old,
                                                            adj_new,
                                                            target_changed_edgelist,
                                                            relu_delta,
                                                            relu_start,
                                                            relu_end, x_tensor, W1, W2)

        summation_to_delta_edge_flag = True

        ceshi_edge_result = np.zeros((adj_new.shape[0], W2.shape[1]))
        for key, value in edge_result.items():
            ceshi_edge_result += value
        true_diff_logits_nonlinear = output_new.detach().numpy() - output_old.detach().numpy()
        for i in range(1):
            if true_diff_logits_nonlinear[i].any() != 0:
                if np.any(abs(ceshi_edge_result[i] - true_diff_logits_nonlinear[i]) > 1e-4):
                    print('key', i, 'test', ceshi_edge_result[i], 'true', true_diff_logits_nonlinear[i])
                    summation_to_delta_edge_flag = False
        print('edge flag', summation_to_delta_edge_flag)

        target_layeredge_result = map_target(layeredge_result, 0)  # in each file, 0 is target node
        target_edge_result = map_target(edge_result, 0)

        if len(target_changed_edgelist) > 5 and len(target_changed_layeredges) > 20:
            global_layeredge_list = select_path_number(args.dataset, len(target_layeredge_result))
            global_edge_list = select_edge_number(args.dataset, len(target_edge_result))
            KL_original = KL_divergence(softmax(output_new[0].detach().numpy()),
                                        softmax(output_old[0].detach().numpy()))

            result_dict = dict()
            result_dict['original KL'] = KL_original
            result_dict['len target_changed_edgelist'] = len(target_changed_edgelist)
            result_dict['len target_layer_edge_list'] = len(target_changed_layeredges)
            result_dict['new prob'] = softmax(output_new[0].detach().numpy()).tolist()
            result_dict['old prob'] = softmax(output_old[0].detach().numpy()).tolist()

            for idx in range(len(global_layeredge_list)):
                select_layeredge = global_layeredge_list[idx]
                print('i', idx, 'select_layeredge', select_layeredge)
                select_layeredges_list = solve_layeredge(select_layeredge, 0, target_layeredge_result,
                                                         target_changed_layeredges,
                                                         output_old.detach().numpy(), output_new)

                evaluate_layeredge_index1, evaluate_layeredge_index2, evaluate_layeredge_weight1, evaluate_layeredge_weight2 = from_layeredges_to_evaulate(
                    select_layeredges_list, edges_weight_old, edges_old, edges_old_dict, adj_old, adj_new)

                evaluate_output = evaluate_model.forward(x_tensor, torch.tensor(evaluate_layeredge_index1), \
                                                         torch.tensor(evaluate_layeredge_index2),
                                                         edge_weight1=torch.tensor(evaluate_layeredge_weight1), \
                                                         edge_weight2=torch.tensor(evaluate_layeredge_weight2))

                KL = KL_divergence(softmax(output_new[0].detach().numpy()),
                                   softmax(evaluate_output[0].detach().numpy()))
                select_layeredges_list_str = map(lambda x: str(x), select_layeredges_list)
                print('select layeredge KL', KL)

                result_dict[str(idx) + ',' + 'select layeredge'] = ",".join(select_layeredges_list_str)
                result_dict[str(idx) + ',' + 'select layeredge' + 'KL'] = KL

            for idx in range(len(global_edge_list)):
                select_edge = global_edge_list[idx]
                print('i', idx, 'select_edge', select_edge)
                select_edges_list = solve_edge(select_edge, 0, target_edge_result, target_changed_edgelist,
                                               output_old.detach().numpy(), output_new)

                # print('select_edges_list ', select_edges_list)
                evaluate_edge_index, evaluate_edge_weight = from_edges_to_evaulate(select_edges_list,
                                                                                   edges_weight_old,
                                                                                   edges_old, edges_old_dict,
                                                                                   adj_old,
                                                                                   adj_new)

                # print('edge weight',evaluate_edge_weight[edges_old_dict[str(7)+','+str(7)]])
                # print('new',adj_new[7,7])
                # print('old',adj_old[7,7])
                #
                # print(len(edges_old[0]),len(edges_new[0]))

                evaluate_edge_output = evaluate_model.forward(x_tensor, torch.tensor(evaluate_edge_index), \
                                                              torch.tensor(evaluate_edge_index),
                                                              edge_weight1=torch.tensor(evaluate_edge_weight), \
                                                              edge_weight2=torch.tensor(evaluate_edge_weight))
                # print('G_t edge', softmax(evaluate_edge_output[0].detach().numpy()))

                # print('ceshi_edge', ceshi_edge[7].detach().numpy())

                KL_edge = KL_divergence(softmax(output_new[0].detach().numpy()),
                                        softmax(evaluate_edge_output[0].detach().numpy()))

                print('select edge KL', KL_edge)

                select_edges_list_str = map(lambda x: str(x), select_edges_list)
                result_dict[str(idx) + ',' + 'select edge'] = ",".join(select_edges_list_str)
                result_dict[str(idx) + ',' + 'select edge' + 'KL'] = KL_edge
            os_path = f'result/{args.dataset}'
            if not os.path.exists(os_path):
                os.makedirs(os_path)
            json_matrix = json.dumps(result_dict)
            with open(f'result/{args.dataset}/{index}.json',
                      'w') as json_file:
                json_file.write(json_matrix)
            print('save success')




