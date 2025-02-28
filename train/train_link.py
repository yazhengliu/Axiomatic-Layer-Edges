import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from  .train_link_utils import SynGraphDataset,split_edge,clear_time,clear_time_UCI
from sklearn.metrics import roc_auc_score,f1_score
import numpy as np
import random
from torch_geometric.utils import train_test_split_edges, negative_sampling
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.int)
    link_labels[:pos_edge_index.size(1)] = 1
    link_labels= link_labels.type(torch.LongTensor)

    return link_labels
def clear_year(time_dict):
    edge_time=dict()
    for key,value in time_dict.items():
        edge_time[key]=value.year
    clear_time = dict()
    for key, value in edge_time.items():
        if (key[1], key[0]) in edge_time.keys():
            clear_time[key] = max(value, edge_time[(key[1], key[0])])
            clear_time[(key[1], key[0])] = max(value, edge_time[(key[1], key[0])])
        if (key[1], key[0]) not in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
    return clear_time
class Net_link_train(torch.nn.Module):
    def __init__(self,data,args):
        self.data=data
        super(Net_link_train, self).__init__()
        # self.features= Parameter(torch.Tensor(data.num_nodes,args.features_size),requires_grad=True)

        self.conv1 = GCNConv(data.num_features, args.hidden,add_self_loops=False,normalize=True,bias=False)
        self.conv2 = GCNConv(args.hidden, args.hidden,add_self_loops=False,normalize=True,bias=False)
        # self.MLP1 = nn.Linear(args.hidden * 2,args.mlp_hidden)
        # self.MLP2 = nn.Linear(args.mlp_hidden, 2)
        self.linear=nn.Linear(args.hidden * 2,2,bias=False)
        self.dropout = args.dropout




    def encode(self,edgeindex):
        # print(type(data.x))
        #
        # print(data.x.type())


        x = self.conv1(self.data.x.to(torch.float32), edgeindex)

        x = x.relu()
        # x = F.dropout(x, self.dropout, training=self.training)
        return self.conv2(x, edgeindex)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)

        h = torch.cat([z[edge_index[0]], z[edge_index[1]]],dim=1)
        h=self.linear(h)

        return h

def train(edgeindex,model,data,optimizer):
    model.train()
    # neg_edge_index = negative_sampling(
    #     edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes-1,
    #     num_neg_samples=data.train_pos_edge_index.size(1),
    #     force_undirected=True,
    # )
    neg_edge_index = negative_sampling(
        edge_index=edgeindex, num_nodes=data.num_nodes-1,
        num_neg_samples=edgeindex.size(1),
        force_undirected=True,
    )

    # print('neg',neg_edge_index)
    optimizer.zero_grad()

    z = model.encode(edgeindex)
    link_logits = model.decode(z, edgeindex, neg_edge_index) #data.train_pos_edge_index
    link_labels = get_link_labels(edgeindex, neg_edge_index) #data.train_pos_edge_index
    # print(link_logits)
    # print(link_labels)
    loss = F.cross_entropy(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(model,data,edgeindex):
    model.eval()
    perfs = []

    for prefix in ["val", "test"]:
        prob_f1 = []
        prob_auc = []
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        z = model.encode(edgeindex)
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        # link_probs = link_logits.sigmoid()
        link_probs=F.softmax(link_logits,dim=1)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        prob_f1.extend(np.argmax( link_probs, axis=1))
        prob_auc.extend(link_probs[:, 1].cpu().numpy())
        # F1=f1_score(link_labels, prob_f1)
        AUC=roc_auc_score(link_labels, prob_auc)

        print("F1-Score:{:.4f}".format(f1_score(link_labels, prob_f1)))
        print("AUC:{:.4f}".format(roc_auc_score(link_labels, prob_auc)))
        perfs.append(AUC)
    return perfs
def train_all(args):

    dataset_name=args.dataset  # amazon_electronics_photo ms_academic_cs
    dataset_dir=f'data/{args.dataset}'
    dataset = SynGraphDataset(dataset_dir, dataset_name)
    modelname = dataset_name
    data = dataset[0]
    #
    data = train_test_split_edges(data)
    # print(len(data.train_pos_edge_index[0]))
    # # print(len(data.train_neg_edge_index[0]))
    # print(len(data.val_pos_edge_index[0]))
    # print(len(data.val_neg_edge_index[0]))
    # print(len(data.test_pos_edge_index[0]))
    # print(len(data.test_neg_edge_index[0]))

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, data = Net_link_train(data,args).to(device), data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    best_val_perf = test_perf = 0
    time_dict = data.time_dict

    data_prefix = f'data/{dataset_name}/'

    if dataset_name == 'UCI':
        clear_time_dict = clear_time_UCI(time_dict)
    else:
        clear_time_dict = clear_time(time_dict)
    for epoch in range(1, args.epochs):

        if dataset_name=='bitcoinalpha' or dataset_name=='bitcoinotc':
            for year1 in range(2013, 2018):
                time_dict = data.time_dict

                edge_index_old = split_edge(2010, year1, 'year', clear_time_dict, data.num_nodes)

                edgeindex = torch.tensor(edge_index_old)
                train_loss = train(edgeindex, model, data, optimizer)
                val_perf, tmp_test_perf = test(model, data, edgeindex)
                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    test_perf = tmp_test_perf
                log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(epoch, train_loss, best_val_perf, test_perf))

                torch.save(model.state_dict(), data_prefix + 'GCN_model_in_' + dataset_name + '.pth')
        elif dataset_name=='UCI':
            for week in range(14, 50):
                time_dict = data.time_dict
                edge_index_old = split_edge(0, week, 'week', clear_time_dict, data.num_nodes)
            #print('edge_index_old',edge_index_old)
            edgeindex = torch.tensor(edge_index_old)
            train_loss = train(edgeindex, model, data, optimizer)
            val_perf, tmp_test_perf = test(model, data, edgeindex)
            if val_perf > best_val_perf:
                best_val_perf = val_perf
                test_perf = tmp_test_perf
            log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_loss, best_val_perf, test_perf))

            torch.save(model.state_dict(), data_prefix + 'GCN_model_in_' + dataset_name + '.pth')

    # model.eval()
    # model.load_state_dict(torch.load(data_prefix + 'GCN_model_in_' + dataset_name + '.pth'))
    #
    # for year1 in range(2010, 2018):
    #     time_dict = data.time_dict
    #     if dataset_name == 'UCI':
    #         clear_time_dict = clear_time_UCI(time_dict)
    #     else:
    #         clear_time_dict = clear_time(time_dict)
    #     edge_index_old = split_edge(2010, year1, 'year', clear_time_dict, data.num_nodes)
    #     edgeindex = torch.tensor(edge_index_old)
    #     test()
    #
    # # for week1 in range(14, 50):
    # #     time_dict = data.time_dict
    # #     if dataset_name == 'UCI':
    # #         clear_time_dict = clear_time_UCI(time_dict)
    # #     else:
    # #         clear_time_dict = clear_time(time_dict)
    # #     edge_index_old = split_edge(0,week1, 'week', clear_time_dict, data.num_nodes)
    # #     edgeindex = torch.tensor(edge_index_old)
    # #     # print(edgeindex)
    # #     test()







