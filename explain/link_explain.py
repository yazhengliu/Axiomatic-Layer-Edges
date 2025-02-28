
import torch
from explain.explain_utils import k_hop_subgraph,split_link_edge,subadj_map_link,rumor_construct_adj_matrix,difference_weight,\
    clear,matrixtodict,subfeaturs,softmax,KL_divergence,findnewpath,reverse_paths,find_target_changed_edges,find_target_changed_layer_edegs,\
    contribution_edge,contribution_layeredge,map_target,mlp_contribution,merge_result,solve_layeredge_link,from_edges_to_evaulate,from_layeredges_to_evaulate,solve_edge_link
import math,copy
import sys
import numpy as np
import os
import json
sys.path.append('..')

from select_args import select_path_number,select_edge_number
from train.train_link_utils import SynGraphDataset,clear_time,clear_time_UCI,split_edge,Net_link
class gen_link_data():
    def __init__(self, dataset,data_path,start1,end1,flag,layernumbers):
        self.dataset=dataset
        self.data_path=data_path
        self.start1 = start1
        self.end1 = end1
        self.flag = flag
        self.layernumbers=layernumbers


    def load_data(self):
        dataset = SynGraphDataset(self.data_path,self.dataset)
        modelname = self.dataset
        data = dataset[0]
        time_dict = data.time_dict
        if self.dataset=='UCI':
            clear_time_dict = clear_time_UCI(time_dict)
        else:
            clear_time_dict = clear_time(time_dict)


        edge_time_result = dict()
        for key,value in clear_time_dict.items():
            if value[1] not in edge_time_result.keys():
                edge_time_result[value[1]]=1
            else:
                edge_time_result[value[1]] =edge_time_result[value[1]]+1

        edge_index_old = split_edge(self.start1, self.end1, self.flag, clear_time_dict,data.num_nodes)



        edge_index_old=torch.tensor(edge_index_old)


        return dataset, edge_index_old,clear_time_dict

    def gen_new_edge(self,target_edge,evaulate_model,edges_all,time_dict,features):
        goal_1 = target_edge[0]
        goal_2 = target_edge[1]
        pos_edge_index = [[goal_1], [goal_2]]
        evaulate_model.eval()
        subset_1_all, edge_index_1_all, _, _ = k_hop_subgraph(
            goal_1, self.layernumbers, edges_all, relabel_nodes=False,
            num_nodes=None)

        subset_2_all, edge_index_2_all, _, _ = k_hop_subgraph(
            goal_2, self.layernumbers, edges_all, relabel_nodes=False,
            num_nodes=None)
        edges_all_dict = dict()
        edge_index_all= [[], []]
        count = 0
        for i in range(len(edge_index_1_all[0])):
            key = str(edge_index_1_all[0][i].item()) + ',' + str(edge_index_1_all[1][i].item())
            edges_all_dict[key] = count
            edge_index_all[0].append(edge_index_1_all[0][i].item())
            edge_index_all[1].append(edge_index_1_all[1][i].item())
            count += 1
        # print('edges_all_dict',edges_all_dict)
        for i in range(len(edge_index_2_all[0])):
            key = str(edge_index_2_all[0][i].item()) + ',' + str(edge_index_2_all[1][i].item())
            if key not in edges_all_dict.keys():
                edges_all_dict[key] = count
                edge_index_all[0].append(edge_index_2_all[0][i].item())
                edge_index_all[1].append(edge_index_2_all[1][i].item())
        all_node_list = list(set(subset_2_all.tolist()).union(set(subset_1_all.tolist())))
        edge_time_result = dict()
        for i in range(len(edge_index_all[0])):
            node1 = edge_index_all[0][i]
            node2 = edge_index_all[1][i]

            if (node1,node2) not in edge_time_result.keys() and (node2,node1) not in edge_time_result.keys():
                edge_time_result[(node1,node2)]=time_dict[(node1,node2)][2]

        sort_edge_time_result = sorted(edge_time_result.items(), key=lambda x: x[1])

        sliding_T = math.floor(len(sort_edge_time_result) / 10)

        edge_index_old = split_link_edge(0, sliding_T * 2, sort_edge_time_result,
                               all_node_list)


        edge_index_new = split_link_edge(sliding_T, sliding_T * 3, sort_edge_time_result,
                               all_node_list)

        submapping, reverse_mapping, map_edge_index_old, map_edge_index_new = subadj_map_link(
            all_node_list, edge_index_old, edge_index_new)
        print('all_node_list', len(all_node_list))
        sub_old = rumor_construct_adj_matrix(map_edge_index_old, len(submapping))
        sub_new = rumor_construct_adj_matrix(map_edge_index_new, len(submapping))
        adj_new_nonzero = sub_new.nonzero()
        adj_old_nonzero = sub_old.nonzero()

        map_edge_old_dict = dict()
        map_edge_weight_old=[]
        for i in range(len(map_edge_index_old[0])):
            map_edge_old_dict[str(map_edge_index_old[0][i]) + ',' + str(map_edge_index_old[1][i])] = i
            map_edge_weight_old.append(sub_old[map_edge_index_old[0][i], map_edge_index_old[1][i]])

        map_edge_new_dict = dict()
        map_edge_weight_new=[]
        for i in range(len(map_edge_index_new[0])):
            map_edge_new_dict[str(map_edge_index_new[0][i]) + ',' + str(map_edge_index_new[1][i])] = i
            map_edge_weight_new.append(sub_new[map_edge_index_new[0][i], map_edge_index_new[1][i]])


        changededgelist = difference_weight(map_edge_index_new, map_edge_index_old, sub_new, sub_old)
        # print(addedgelist)
        changededgelist = clear(changededgelist)

        graph_old = matrixtodict(adj_old_nonzero)
        graph_new = matrixtodict(adj_new_nonzero)

        graph_all = copy.deepcopy(graph_old)
        # print('graph_all', graph_all)
        for edge in changededgelist:
            if edge[1] not in graph_all[edge[0]]:
                graph_all[edge[0]].append(edge[1])
            if edge[0] not in graph_all[edge[1]]:
                graph_all[edge[1]].append(edge[0])

        map_edge_old_dict_reverse = dict()
        for key, value in map_edge_old_dict.items():
            map_edge_old_dict_reverse[value] = key

        map_edge_new_dict_reverse = dict()
        for key, value in map_edge_new_dict.items():
            map_edge_new_dict_reverse[value] = key

        sub_features = subfeaturs(features, reverse_mapping)
        sub_features = torch.tensor(sub_features)
        sub_features = sub_features.to(torch.float32)


        map_edge_index_old = torch.tensor(map_edge_index_old)
        map_edge_weight_old = torch.tensor(map_edge_weight_old)
        map_edge_weight_old = map_edge_weight_old.to(torch.float32)
        map_edge_index_new = torch.tensor(map_edge_index_new)

        map_edge_weight_new = torch.tensor(map_edge_weight_new)

        map_edge_weight_new = map_edge_weight_new.to(torch.float32)

        return sub_features, sub_old, sub_new, map_edge_index_old, map_edge_index_new, graph_old, graph_new, graph_all, changededgelist, \
            map_edge_weight_old, map_edge_weight_new, map_edge_old_dict, map_edge_new_dict, submapping,map_edge_old_dict_reverse, map_edge_new_dict_reverse






    def gen_model(self,data,hidden):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model= Net_link(data.num_features,hidden).to(device)
        model.eval()
        data_prefix = self.data_path+'/'
        model.load_state_dict(torch.load(data_prefix+ 'GCN_model_in_'+self.dataset+'.pth'))

        W1 = model.state_dict()['conv1.lin.weight'].t()
        W2 = model.state_dict()['conv2.lin.weight'].t()

        W3=model.state_dict()['linear.weight'].t()

        # print(model.state_dict().keys())
        return model,W1,W2,W3


    def gen_parameters(self,model, features,edges_old_tensor,edges_new_tensor,edgeweight1,edgeweight2):
        model.eval()
        nonlinear_start_layer1, nonlinear_relu_start_layer1 = model.back(features, edges_old_tensor, edges_old_tensor,edgeweight1, edgeweight1)
        nonlinear_end_layer1, nonlinear_relu_end_layer1 = model.back(features, edges_new_tensor, edges_new_tensor,
                                                                     edgeweight2, edgeweight2)

        # print('nonlinear_start_layer1',nonlinear_start_layer1.shape)
        # print('nonlinear_end_layer1', nonlinear_end_layer1.shape)
        #
        # print('nonlinear_start_layer1', nonlinear_relu_start_layer1.shape)
        # print('nonlinear_end_layer1',  nonlinear_relu_end_layer1.shape)

        relu_delta = torch.where((nonlinear_end_layer1 - nonlinear_start_layer1) != 0,
                                 (nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1) / (
                                         nonlinear_end_layer1 - nonlinear_start_layer1),
                                 torch.zeros_like((nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1)))
        relu_end = torch.where((nonlinear_end_layer1) != 0, nonlinear_relu_end_layer1 / nonlinear_end_layer1,
                               torch.zeros_like(nonlinear_end_layer1))
        relu_start = torch.where((nonlinear_start_layer1) != 0, nonlinear_relu_start_layer1 / nonlinear_start_layer1,
                                 torch.zeros_like(nonlinear_start_layer1))
        return relu_delta, relu_end, relu_start
def explain_link(args):
    modelname = args.dataset
    if args.dataset == 'UCI':
        flag = 'week'
        start_time=0
        end_time = 200
    else:
        flag = 'month'
        start_time = 0
        end_time = 200
    data_path=f'data/{modelname}'
    layernumbers =args.layernumbers

    dynamic_data = gen_link_data(args.dataset, data_path, start_time,end_time, flag, layernumbers)

    data, edges_old, clear_time_dict = dynamic_data.load_data()
    features = data.data.x
    features = features.to(torch.float32)

    hidden=4

    model, W1, W2, W3 = dynamic_data.gen_model(data,hidden)

    target_edge_list = [[53,562]] #explain edge

    for target_edge in target_edge_list:
        sub_features, sub_old, sub_new, map_edge_index_old, map_edge_index_new, sub_graph_old, sub_graph_new, sub_graph_all, changededgelist, \
            map_edge_weight_old, map_edge_weight_new, map_edge_old_dict, map_edge_new_dict, submapping, map_edge_old_dict_reverse, map_edge_new_dict_reverse = dynamic_data.gen_new_edge(
            target_edge, model, edges_old, clear_time_dict, features)
        goal_1 = submapping[target_edge[0]]
        goal_2 = submapping[target_edge[1]]

        pos_edge_index = [[goal_1], [goal_2]]
        pos_edge_index = torch.tensor(pos_edge_index)

        print('len(changededgelist)', len(changededgelist))

        encode_logits_old = model.encode(sub_features, map_edge_index_old, map_edge_index_old, map_edge_weight_old,
                                         map_edge_weight_old)
        decode_logits_old = model.decode(encode_logits_old, pos_edge_index).view(-1)
        decode_logits_old_numpy = decode_logits_old.detach().numpy()
        encode_logits_new = model.encode(sub_features, map_edge_index_new, map_edge_index_new, map_edge_weight_new,
                                         map_edge_weight_new)
        decode_logits_new = model.decode(encode_logits_new, pos_edge_index).view(-1)
        decode_logits_new_numpy = decode_logits_new.detach().numpy()

        relu_delta, relu_end, relu_start = dynamic_data.gen_parameters(model, sub_features, map_edge_index_old,
                                                                       map_edge_index_new,
                                                                       map_edge_weight_old, map_edge_weight_new,
                                                                       )
        G_new = softmax(decode_logits_new_numpy)
        G_old = softmax(decode_logits_old_numpy)

        KL_original = KL_divergence(G_new,
                                    G_old)

        changed_paths_goal1 = findnewpath(changededgelist, sub_graph_all, layernumbers, goal_1)
        target_path_1 = reverse_paths(changed_paths_goal1)

        changed_paths_goal2 = findnewpath(changededgelist, sub_graph_all, layernumbers, goal_2)
        target_path_2 = reverse_paths(changed_paths_goal2)

        target1_changed_edgelist = find_target_changed_edges(target_path_1, changededgelist)
        target1_layer_edge_list=find_target_changed_layer_edegs(target1_changed_edgelist,
                                                                                  args.layernumbers)


        target2_changed_edgelist = find_target_changed_edges(target_path_1, changededgelist)
        target2_layer_edge_list = find_target_changed_layer_edegs(target2_changed_edgelist,
                                                                  args.layernumbers)

        _, _, test_layeredge_result_1 = contribution_layeredge(target_path_1,
                                                                         sub_old,
                                                                         sub_new,
                                                                         target1_changed_edgelist,
                                                                         relu_delta,
                                                                         relu_start,
                                                                         relu_end, sub_features, W1, W2)

        target1_layeredge_result = map_target(test_layeredge_result_1, goal_1)

        _, _, test_layeredge_result_2 = contribution_layeredge(target_path_2,
                                                                         sub_old,
                                                                         sub_new,
                                                                         target2_changed_edgelist,
                                                                         relu_delta,
                                                                         relu_start,
                                                                         relu_end, sub_features, W1, W2)
        target2_layeredge_result = map_target(test_layeredge_result_2, goal_2)

        final_target1_layeredge_result = mlp_contribution(target1_layeredge_result, W3[:encode_logits_new.shape[1]])

        final_target2_layeredge_result = mlp_contribution(target2_layeredge_result, W3[encode_logits_new.shape[1]:])

        target_layeredge_result=merge_result(final_target1_layeredge_result,final_target2_layeredge_result)
        summation_to_delta_layeredge_flag = True
        ceshi_edge_result = np.zeros((W3.shape[1]))
        for key, value in target_layeredge_result.items():
            ceshi_edge_result += value

        true_diff_logits_nonlinear = decode_logits_new_numpy - \
                                     decode_logits_old_numpy
        if true_diff_logits_nonlinear.any() != 0:
            if np.any(abs(ceshi_edge_result - true_diff_logits_nonlinear) > 1e-4):
                print('key', 'test', ceshi_edge_result, 'true', true_diff_logits_nonlinear)
                summation_to_delta_layeredge_flag = False
        print('target layeredge flag', summation_to_delta_layeredge_flag)

        _, _, edge_result_1 = contribution_edge(target_path_1,
                                                               sub_old,
                                                               sub_new,
                                                               target1_changed_edgelist,
                                                               relu_delta,
                                                               relu_start,
                                                               relu_end, sub_features, W1, W2)
        _, _, edge_result_2 = contribution_edge(target_path_2,
                                                               sub_old,
                                                               sub_new,
                                                               target2_changed_edgelist,
                                                               relu_delta,
                                                               relu_start,
                                                               relu_end, sub_features, W1, W2)
        target1_edge_result = map_target(edge_result_1, goal_1)
        target2_edge_result = map_target(edge_result_2, goal_2)

        final_target1_edge_result = mlp_contribution(target1_edge_result, W3[:encode_logits_new.shape[1]])
        final_target2_edge_result = mlp_contribution(target2_edge_result, W3[encode_logits_new.shape[1]:])

        target_edge_result = merge_result(final_target1_edge_result,final_target2_edge_result)

        summation_to_delta_edge_flag = True

        ceshi_edge_result = np.zeros((W3.shape[1]))
        for key, value in target_edge_result.items():
            ceshi_edge_result += value

        if true_diff_logits_nonlinear.any() != 0:
            if np.any(abs(ceshi_edge_result - true_diff_logits_nonlinear) > 1e-4):
                print('key', 'test', ceshi_edge_result, 'true', true_diff_logits_nonlinear)
                summation_to_delta_edge_flag  = False
        print('target edge flag', summation_to_delta_edge_flag)

        target_layer_edge_list = []
        for layeredge in target1_layer_edge_list:
            if layeredge not in target_layer_edge_list:
                target_layer_edge_list.append(layeredge)
        for layeredge in target2_layer_edge_list:
            if layeredge not in target_layer_edge_list:
                target_layer_edge_list.append(layeredge)

        target_changed_edgelist = []
        for edge in target1_changed_edgelist:
            if edge not in target_changed_edgelist:
                target_changed_edgelist.append(edge)
        for edge in target2_changed_edgelist:
            if edge not in target_changed_edgelist:
                target_changed_edgelist.append(edge)

        global_layeredge_list = select_path_number(args.dataset,
                                            len(target_layer_edge_list))

        global_edge_list = select_edge_number(args.dataset,
                                        len(target_changed_edgelist))



        if len(target_changed_edgelist) > 10 and len(target_layer_edge_list) > 40:
            result_dict=dict()
            result_dict['original KL'] = KL_original
            result_dict['len target_changed_edgelist'] = len(target_changed_edgelist)
            result_dict['len target_layer_edge_list'] = len(target_layer_edge_list)

            result_dict['new prob'] = softmax(G_new).tolist()
            result_dict['old prob'] = softmax(G_old).tolist()

            for idx in range(len(global_layeredge_list)):
                select_layeredge = global_layeredge_list[idx]

                print('i', idx, 'select_layeredge', select_layeredge)
                select_layeredges_list = solve_layeredge_link(select_layeredge,
                                                  target_layeredge_result, target_layer_edge_list,
                                                  decode_logits_old_numpy,
                                                  decode_logits_new_numpy)


                evaluate_layeredge_index1, evaluate_layeredge_index2, evaluate_layeredge_weight1, evaluate_layeredge_weight2 = from_layeredges_to_evaulate(
                    select_layeredges_list, map_edge_weight_old, map_edge_index_old.tolist(), map_edge_old_dict, sub_old,
                    sub_new)


                evaluate_encode = model.encode(sub_features, torch.tensor(evaluate_layeredge_index1), \
                                               torch.tensor(evaluate_layeredge_index2),
                                               edge_weight1=torch.tensor(evaluate_layeredge_weight1), \
                                               edge_weight2=torch.tensor(evaluate_layeredge_weight2))
                evaluate_decode = model.decode(evaluate_encode, torch.tensor([[goal_1], [goal_2]])).squeeze()
                #print('G_t', softmax(evaluate_decode.detach().numpy()))
                KL = KL_divergence(softmax(decode_logits_new_numpy),
                                   softmax(evaluate_decode.detach().numpy()))

                select_layeredges_list_str = map(lambda x: str(x), select_layeredges_list)
                print('select layeredge KL', KL)

                result_dict[str(idx) + ',' + 'select layeredge'] = ",".join(select_layeredges_list_str)
                result_dict[str(idx) + ',' + 'select layeredge' + 'KL'] = KL

            for idx in range(len(global_edge_list)):
                select_edge = global_edge_list[idx]
                print('i', idx, 'select_edge', select_edge)
                select_edges_list = solve_edge_link(select_edge, target_edge_result, target_changed_edgelist,
                                                  decode_logits_old_numpy,
                                                  decode_logits_new_numpy)

                # print('select_edges_list ', select_edges_list)
                evaluate_edge_index, evaluate_edge_weight = from_edges_to_evaulate(select_edges_list,
                                                                                   map_edge_weight_old,
                                                                                   map_edge_index_old.tolist(),
                                                                                   map_edge_old_dict, sub_old, sub_new)

                evaluate_edge_encode = model.encode(sub_features, torch.tensor(evaluate_edge_index), \
                                                    torch.tensor(evaluate_edge_index),
                                                    edge_weight1=torch.tensor(evaluate_edge_weight), \
                                                    edge_weight2=torch.tensor(evaluate_edge_weight))
                evaluate_edge_decode = model.decode(evaluate_edge_encode, torch.tensor([[goal_1], [goal_2]])).squeeze()

                KL_edge = KL_divergence(
                    softmax(decode_logits_new_numpy),
                    softmax(
                        evaluate_edge_decode.detach().numpy()))

                print('select edge KL', KL_edge)

                select_edges_list_str = map(lambda x: str(x), select_edges_list)
                result_dict[str(idx) + ',' + 'select edge'] = ",".join(select_edges_list_str)
                result_dict[str(idx) + ',' + 'select edge' + 'KL'] = KL_edge

            os_path = f'result/{args.dataset}'
            if not os.path.exists(os_path):
                os.makedirs(os_path)

            json_matrix = json.dumps(result_dict)
            with open(f'result/{args.dataset}/{target_edge[0]}_{target_edge[1]}.json',
                      'w') as json_file:
                json_file.write(json_matrix)
            print('save success')




























