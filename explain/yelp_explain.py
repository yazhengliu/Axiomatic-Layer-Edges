import json
import random
import pickle
import torch
import numpy as np
import argparse
import scipy.sparse as sp
import math
import os
import copy
import cvxpy as cvx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from  .explain_utils import k_hop_subgraph,subadj_map,subfeaturs,rumor_construct_adj_matrix, matrixtodict,clear,findnewpath,reverse_paths,KL_divergence,softmax,split,difference_weight,\
    contribution_edge,contribution_layeredge,find_target_changed_edges,find_target_changed_layer_edegs,map_target,from_edges_to_evaulate,from_layeredges_to_evaulate,solve_edge,solve_layeredge

import sys
sys.path.append('..')
from select_args import select_path_number,select_edge_number
from train.train_yelp_utils import read_data,read_user_prod,feature_matrix,onehot_label,construct_edge,sparse_mx_to_torch_sparse_tensor,GCN, GCN_test
from select_args import select_path_number,select_edge_number
class gen_Yelp_data():
    def __init__(self,dataset,start,end):
        self.dataset=dataset
        self.start=start
        self.end=end
    def gen_data(self):
        target_domain = self.dataset  # test
        data_prefix = f'data/{self.dataset}/'
        with open(data_prefix + target_domain + '_features.pickle', 'rb') as f:
            raw_features = pickle.load(f)
        with open(data_prefix + 'ground_truth_' + target_domain, 'rb') as f:
            review_ground_truth = pickle.load(f)

        with open(data_prefix + f'{self.dataset}_split_data.pickle', 'rb') as f:
            rev_time = pickle.load(f)
        # print('rev_time',rev_time)

        count_time=dict()
        for it, r_id in enumerate(review_ground_truth.keys()):
            if rev_time[r_id][1] not in count_time.keys():
                count_time[rev_time[r_id][1]]=1
            else:
                count_time[rev_time[r_id][1]] =count_time[rev_time[r_id][1]]+1
        print('count_time',count_time)

        train_ratio = 0.5
        val_ratio = 0.2
        train_rev = read_data('train', 'review', target_domain, train_ratio, val_ratio)
        val_rev = read_data('val', 'review', target_domain, train_ratio, val_ratio)
        test_rev = read_data('test', 'review', target_domain, train_ratio, val_ratio)
        # print('train_rev',train_rev)
        train_user, train_prod = read_user_prod(train_rev)
        val_user, val_prod = read_user_prod(val_rev)
        test_user, test_prod = read_user_prod(test_rev)
        # print('train_user',train_user)

        portion_train = train_rev + train_user
        portion_val = val_rev + val_user
        portion_test = test_rev + test_user
        # print('portion_train',portion_train)
        list_idx, features, nums = feature_matrix(raw_features, portion_train, portion_val, portion_test)

        labels, user_ground_truth = onehot_label(review_ground_truth, list_idx)
        #print(labels)
        idx_map = {j: i for i, j in enumerate(list_idx)}
        rev_list = []
        rev_label = []
        for key, value in review_ground_truth.items():
            rev_list.append(idx_map[key])
            rev_label.append(value)
        print('rev_list', len(rev_list))

        user_list = []
        user_label = []
        for key, value in user_ground_truth.items():
            user_list.append(idx_map[key])
            user_label.append(value)
        print('user list', len(user_list))
        # print('idx_map',idx_map)
        idx_map_reverse={v: k for k, v in idx_map.items()}
        # print(idx_map_reverse)


        edges_old = construct_edge(review_ground_truth, idx_map, labels, rev_time, self.start, self.end, 'month')


        return features, nums,user_list,user_label,rev_list,rev_label,edges_old,labels,rev_time,idx_map_reverse



    def gen_adj(self,goal,edges_old,clean_features,rev_time,idx_map_reverse,layernumbers):
        subset_all, edge_index_all, _, _ = k_hop_subgraph(
            goal, layernumbers, edges_old, relabel_nodes=False,
            num_nodes=None)
        count=0

        edge_time_result=dict()
        for i in range(len(edge_index_all[0])):
            node1=edge_index_all[0][i].item()
            node2 = edge_index_all[1][i].item()

            if idx_map_reverse[node1] in rev_time.keys():
                # print(rev_time[idx_map_reverse[node1]])
                if str(node1)+','+str(node2) not in edge_time_result.keys() and str(node2)+','+str(node1) not in edge_time_result.keys() :
                    edge_time_result[str(node1) + ',' + str(node2)] = rev_time[idx_map_reverse[node1]][2]
                    count += 1

            if idx_map_reverse[node2] in rev_time.keys():
                # print(rev_time[idx_map_reverse[node2]])
                if str(node1) + ',' + str(node2) not in edge_time_result.keys() and str(node2) + ',' + str(
                        node1) not in edge_time_result.keys():
                    edge_time_result[str(node1) + ',' + str(node2)] = rev_time[idx_map_reverse[node2]][2]
                    count += 1

        sort_edge_time_result=sorted(edge_time_result.items(), key=lambda x: x[1])


        sliding_T = math.floor(len(sort_edge_time_result) / 4)

        edge_index_old=split(0,sliding_T*3,sort_edge_time_result,
                                 subset_all)

        edge_index_new = split( sliding_T , len(sort_edge_time_result),sort_edge_time_result,
                               subset_all)
        # print(edge_index_new[0])
        # print(edge_index_new[1])

        submapping, reverse_mapping, map_edge_index_old, map_edge_index_new = subadj_map(
            subset_all, edge_index_old,  edge_index_new)

        sub_features = subfeaturs(clean_features, reverse_mapping)
        sub_old = rumor_construct_adj_matrix(map_edge_index_old, len(submapping))
        sub_new = rumor_construct_adj_matrix(map_edge_index_new, len(submapping))

        adj_new_nonzero = sub_new.nonzero()
        adj_old_nonzero = sub_old.nonzero()

        map_edge_weight_old = []
        edges_old_dict = dict()

        for i in range(len(map_edge_index_old[0])):
            map_edge_weight_old.append(sub_old[map_edge_index_old[0][i], map_edge_index_old[1][i]])
            edges_old_dict[str(map_edge_index_old[0][i]) + ',' + str(map_edge_index_old[1][i])] = i
        map_edge_weight_old = torch.tensor(map_edge_weight_old)

        map_edge_weight_new = []
        edges_new_dict = dict()

        for i in range(len(map_edge_index_new[0])):
            map_edge_weight_new.append(sub_new[map_edge_index_new[0][i], map_edge_index_new[1][i]])
            edges_new_dict[str(map_edge_index_new[0][i]) + ',' + str(map_edge_index_new[1][i])] = i
        map_edge_weight_new= torch.tensor(map_edge_weight_new)

        changededgelist = difference_weight(map_edge_index_new, map_edge_index_old, sub_new, sub_old)
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

        sub_features = torch.tensor(sub_features)
        sub_features = sub_features.to(torch.float32)

        map_edge_weight_old = torch.tensor(map_edge_weight_old)
        map_edge_weight_old = map_edge_weight_old.to(torch.float32)

        map_edge_weight_new = torch.tensor(map_edge_weight_new)
        map_edge_weight_new = map_edge_weight_new.to(torch.float32)

        map_edge_index_old = torch.tensor(map_edge_index_old)
        map_edge_index_new = torch.tensor(map_edge_index_new)

        return sub_features,sub_old, sub_new, map_edge_index_old, map_edge_index_new, graph_old, graph_new, graph_all, changededgelist, map_edge_weight_old, map_edge_weight_new, edges_old_dict, edges_new_dict,submapping,edges_old_dict_reverse,edges_new_dict_reverse


    def gen_parameters(self,features,edges_old_tensor,edges_new_tensor,edgeweight1,edgeweight2,model):

        nonlinear_start_layer1, nonlinear_relu_start_layer1 = model.back(features, edges_old_tensor, edges_old_tensor,
                                                                         edgeweight1, edgeweight1)
        nonlinear_end_layer1, nonlinear_relu_end_layer1 = model.back(features, edges_new_tensor, edges_new_tensor,
                                                                     edgeweight2, edgeweight2)


        relu_delta = torch.where((nonlinear_end_layer1 - nonlinear_start_layer1) != 0,
                                 (nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1) / (
                                             nonlinear_end_layer1 - nonlinear_start_layer1),
                                 torch.zeros_like((nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1)))
        relu_end = torch.where((nonlinear_end_layer1) != 0, nonlinear_relu_end_layer1 / nonlinear_end_layer1,
                               torch.zeros_like(nonlinear_end_layer1))
        relu_start = torch.where((nonlinear_start_layer1) != 0, nonlinear_relu_start_layer1 / nonlinear_start_layer1,
                                 torch.zeros_like(nonlinear_start_layer1))
        return relu_delta,relu_end,relu_start







    def gen_model(self,features, nums):
        parser = argparse.ArgumentParser()
        parser.add_argument('--hidden', type=int, default=16,
                            help='Number of hidden units.')
        parser.add_argument('--train_ratio', type=float, default=0.5)
        parser.add_argument('--val_ratio', type=float, default=0.2)
        parser.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate (1 - keep probability).')

        args = parser.parse_args()

        model = GCN(nfeat=32,
                    nhid=args.hidden,
                    nclass=2,
                    dropout=args.dropout)

        model.eval()
        data_prefix=f'data/{self.dataset}/'
        model.load_state_dict(torch.load(data_prefix + 'GCN_model_in_' + self.dataset + '.pth'))
        x_tensor=model.feature(features, nums)

        W1 = model.state_dict()['gc1.lin.weight'].t()
        W2 = model.state_dict()['gc2.lin.weight'].t()



        return model,x_tensor,W1,W2

    def gen_evaluate_model(self, model):
        parser = argparse.ArgumentParser()

        parser.add_argument('--hidden', type=int, default=16,
                            help='Number of hidden units.')
        parser.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--train_ratio', type=float, default=0.5)
        parser.add_argument('--val_ratio', type=float, default=0.2)

        args = parser.parse_args()

        model_gnn = GCN_test(nfeat=32,
                    nhid=args.hidden,
                    nclass=2,
                    dropout=args.dropout)
        model_gnn.eval()

        model_dict = model_gnn.state_dict()
        #print('model.state_dict()',model.state_dict())
        model_dict['conv1.lin.weight'] = model.state_dict()['gc1.lin.weight']
        model_dict['conv2.lin.weight'] = model.state_dict()['gc2.lin.weight']
        model_gnn.load_state_dict(model_dict)
        return model_gnn

def explain_yelp(args):
    start_time=0
    end_time=200
    Yelp_data = gen_Yelp_data(args.dataset, start_time, end_time)
    features, nums, user_list, user_label, rev_list, rev_label, edges_old, labels, rev_time, idx_map_reverse = Yelp_data.gen_data()
    model, x_tensor, W1, W2 = Yelp_data.gen_model(features, nums)

    evaluate_model = Yelp_data.gen_evaluate_model(model)
    num_rev = 40
    num_user = 40
    random.seed(42)
    target_rev_list = random.sample(rev_list, num_rev)
    target_use_list = random.sample(user_list, num_user)
    test_goal_list=target_rev_list+target_use_list

    print(test_goal_list)

    for test_node in [test_goal_list[0]]:
        print('test_node',test_node)
        sub_features, sub_adj_old, sub_adj_new, sub_edge_index_old, sub_edge_index_new, sub_graph_old, sub_graph_new, sub_graph_all, changededgelist, sub_edge_weight_old, sub_edge_weight_new, sub_edges_old_dict, sub_edges_new_dict, submapping, sub_edges_old_dict_reverse, sub_edges_new_dict_reverse = Yelp_data.gen_adj(
            test_node, torch.tensor(edges_old), x_tensor.detach().numpy(), rev_time, idx_map_reverse,args.layernumbers)

        sub_adj_old = sparse_mx_to_torch_sparse_tensor(sub_adj_old)
        sub_adj_new = sparse_mx_to_torch_sparse_tensor(sub_adj_new)

        changed_paths=findnewpath(changededgelist, sub_graph_all, args.layernumbers, submapping[test_node])

        target_changed_path = reverse_paths(changed_paths)

        target_changed_edgelist = find_target_changed_edges(target_changed_path, changededgelist)

        target_changed_layeredges = find_target_changed_layer_edegs(target_changed_edgelist,
                                                                    args.layernumbers)

        print('changededgelist', len(changededgelist))
        print('target_changed_edgelist', len(target_changed_edgelist))
        print('target_layer_edge_list', len(target_changed_layeredges))

        if len(target_changed_edgelist) > 5 and len(target_changed_layeredges) > 20:
            evaluate_model.eval()
            model.eval()

            relu_delta, relu_end, relu_start = Yelp_data.gen_parameters(sub_features, sub_edge_index_old,
                                                                        sub_edge_index_new,
                                                                        sub_edge_weight_old, sub_edge_weight_new,
                                                                        evaluate_model)


            output_new = \
            evaluate_model.forward(sub_features, sub_edge_index_new, sub_edge_index_new, sub_edge_weight_new,
                                   sub_edge_weight_new)
            output_old = \
            evaluate_model.forward(sub_features, sub_edge_index_old, sub_edge_index_old, sub_edge_weight_old,
                                   sub_edge_weight_old)
            KL_original = KL_divergence(softmax(output_new[submapping[test_node]].detach().numpy()),
                                        softmax(output_old[submapping[test_node]].detach().numpy()))

            _, _, layeredge_result = contribution_layeredge(target_changed_path,
                                                                                     sub_adj_old,
                                                                                     sub_adj_new,
                                                                                     target_changed_edgelist,
                                                                                     relu_delta,
                                                                                     relu_start,
                                                                                     relu_end, sub_features, W1, W2)
            summation_to_delta_layeredge_flag = True
            ceshi_edge_result = np.zeros((sub_adj_old.shape[0], W2.shape[1]))
            for key, value in layeredge_result.items():
                ceshi_edge_result += value
            #print('ceshi_edge_result', ceshi_edge_result[submapping[test_node]])
            true_diff_logits_nonlinear = output_new[submapping[test_node]].detach().numpy() - output_old[submapping[test_node]].detach().numpy()
            #print('true_diff_logits_nonlinear', true_diff_logits_nonlinear)
            if true_diff_logits_nonlinear.any() != 0:
                if np.any(abs(ceshi_edge_result[submapping[test_node]] - true_diff_logits_nonlinear) > 1e-4):
                    #print('key', 'test', ceshi_edge_result, 'true', true_diff_logits_nonlinear)
                    summation_to_delta_layeredge_flag = False
            print('layeredge flag', summation_to_delta_layeredge_flag)

            _, _, edge_result = contribution_edge(target_changed_path,
                                                                 sub_adj_old,
                                                                 sub_adj_new,
                                                                 target_changed_edgelist,
                                                                 relu_delta,
                                                                 relu_start,
                                                                 relu_end, sub_features, W1, W2)
            summation_to_delta_edge_flag = True

            ceshi_edge_result = np.zeros((sub_adj_old.shape[0], W2.shape[1]))
            for key, value in edge_result.items():
                ceshi_edge_result += value
            # print('ceshi_edge_result', ceshi_edge_result)
            true_diff_logits_nonlinear = output_new[submapping[test_node]].detach().numpy() - output_old[submapping[test_node]].detach().numpy()
            if true_diff_logits_nonlinear.any() != 0:
                if np.any(abs(ceshi_edge_result[submapping[test_node]] - true_diff_logits_nonlinear) > 1e-4):
                    print('key', 'test', ceshi_edge_result[submapping[test_node]], 'true', true_diff_logits_nonlinear)
                    summation_to_delta_edge_flag  = False
            print('edge flag', summation_to_delta_edge_flag)

            target_layeredge_result = map_target(layeredge_result, submapping[test_node])
            target_edge_result = map_target(edge_result, submapping[test_node])

            global_layeredge_list = select_path_number(args.dataset,  len(target_changed_edgelist))

            global_edge_list = select_edge_number(args.dataset, len(target_changed_edgelist))

            result_dict = dict()
            result_dict['original KL'] = KL_original
            result_dict['len target_changed_edgelist'] = len(target_changed_edgelist)
            result_dict['len target_layer_edge_list'] = len(target_changed_layeredges)

            result_dict['new prob'] = softmax(output_new[submapping[test_node]].detach().numpy()).tolist()
            result_dict['old prob'] = softmax(output_old[submapping[test_node]].detach().numpy()).tolist()

            for idx in range(len(global_layeredge_list)):
                select_layeredge = global_layeredge_list[idx]

                print('i', idx, 'select_layeredge', select_layeredge)
                select_layeredges_list = solve_layeredge(select_layeredge, submapping[test_node],
                                                  target_layeredge_result, target_changed_layeredges,
                                                  output_old.detach(), output_new.detach())

                #
                evaluate_layeredge_index1, evaluate_layeredge_index2, evaluate_layeredge_weight1, evaluate_layeredge_weight2 = from_layeredges_to_evaulate(
                select_layeredges_list, sub_edge_weight_old, sub_edge_index_old.tolist(), sub_edges_old_dict, sub_adj_old, sub_adj_new)

                evaluate_output = evaluate_model.forward(sub_features, torch.tensor(evaluate_layeredge_index1), \
                                                         torch.tensor(evaluate_layeredge_index2),
                                                         edge_weight1=torch.tensor(evaluate_layeredge_weight1), \
                                                         edge_weight2=torch.tensor(evaluate_layeredge_weight2))


                KL = KL_divergence(softmax(output_new[submapping[test_node]].detach().numpy()),
                                   softmax(evaluate_output[submapping[test_node]].detach().numpy()))
                select_layeredges_list_str = map(lambda x: str(x), select_layeredges_list)
                print('select layeredge KL', KL)

                result_dict[str(idx) + ',' + 'select layeredge'] = ",".join(select_layeredges_list_str)
                result_dict[str(idx) + ',' + 'select layeredge' + 'KL'] = KL

            for idx in range(len(global_edge_list)):

                select_edge = global_edge_list[idx]
                print('i', idx, 'select_edge', select_edge)
                select_edges_list = solve_edge(select_edge, submapping[test_node],target_edge_result, target_changed_edgelist,
                                                  output_old.detach(), output_new.detach())

                # print('select_edges_list ', select_edges_list)
                evaluate_edge_index, evaluate_edge_weight = from_edges_to_evaulate(select_edges_list,
                                                                                   sub_edge_weight_old,
                                                                                   sub_edge_index_old.tolist(), sub_edges_old_dict,
                                                                                   sub_adj_old, sub_adj_new)

                evaluate_edge_output = evaluate_model.forward(sub_features, torch.tensor(evaluate_edge_index), \
                                                              torch.tensor(evaluate_edge_index),
                                                              edge_weight1=torch.tensor(evaluate_edge_weight), \
                                                              edge_weight2=torch.tensor(evaluate_edge_weight))
                # print('G_t edge', softmax(evaluate_edge_output[0].detach().numpy()))

                # print('ceshi_edge', ceshi_edge[7].detach().numpy())

                KL_edge = KL_divergence(softmax(output_new[submapping[test_node]].detach().numpy()),
                                        softmax(
                                            evaluate_edge_output[submapping[test_node]].detach().numpy()))

                print('select edge KL', KL_edge)

                select_edges_list_str = map(lambda x: str(x), select_edges_list)
                result_dict[str(idx) + ',' + 'select edge'] = ",".join(select_edges_list_str)
                result_dict[str(idx) + ',' + 'select edge' + 'KL'] = KL_edge

            os_path = f'result/{args.dataset}'
            if not os.path.exists(os_path):
                os.makedirs(os_path)

            json_matrix = json.dumps(result_dict)
            with open(f'result/{args.dataset}/{test_node}.json',
                      'w') as json_file:
                json_file.write(json_matrix)
            print('save success')













