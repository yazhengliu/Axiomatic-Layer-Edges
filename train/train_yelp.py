from __future__ import division
from __future__ import print_function
import torch.nn as nn
import time
import argparse
import numpy as np
import pickle
import scipy.sparse as sp
import random

import torch
import torch.nn.functional as F
import torch.optim as optim


from .train_yelp_utils import GCN
from .train_yelp_utils import read_data,read_user_prod,feature_matrix,onehot_label,construct_adj_matrix,\
    sparse_mx_to_torch_sparse_tensor,accuracy,auc_score

def train_all(args):


    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio

    target_domain = args.dataset  # test
    source_domain = args.dataset
    data_prefix = f'data/{args.dataset}/'
    with open(data_prefix + target_domain + '_features.pickle', 'rb') as f:
        raw_features = pickle.load(f)

    with open(data_prefix + 'ground_truth_' + target_domain, 'rb') as f:
        review_ground_truth = pickle.load(f)

    with open(data_prefix + 'messages_' + target_domain, 'rb') as f:
        messages = pickle.load(f)

    with open(data_prefix + f'{args.dataset}_split_data.pickle', 'rb') as f:
        rev_time = pickle.load(f)
    # for key,value in raw_features.items():
    #     print('key',key)
    #     # print('value',value)
    # print('features',raw_features['u13258'])
    # print('review_ground_truth',review_ground_truth)
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
    print(labels)
    idx_map = {j: i for i, j in enumerate(list_idx)}
    labels = torch.LongTensor(np.where(labels)[1])
    print(labels)
    # # year3=2008
    max_month = 0
    for it, r_id in enumerate(review_ground_truth.keys()):
        if rev_time[r_id][1] >= max_month:
            max_month = rev_time[r_id][1]
    print('max month', max_month)
    # print('rev_time',rev_time)
    # # print(labels)
    #
    # # adj_new = utils.construct_adj_matrix(review_ground_truth, idx_map, labels,rev_time,year2,year3)
    # # # adj = utils.normalize(adj + sp.eye(adj.shape[0]))
    # # adj_new=adj_new+sp.eye(adj_new.shape[0])
    # # print(adj_0)
    # # adj_new=adj_new.todense()
    # # adj_0=adj_0+sp.eye(adj.shape[0])
    # # adj_new = utils.sparse_mx_to_torch_sparse_tensor(adj_new)
    #
    idx_train = torch.LongTensor(range(nums[-1][0]))
    idx_val = torch.LongTensor(range(nums[-1][0], nums[-1][1]))
    idx_test = torch.LongTensor(range(nums[-1][1], nums[-1][2]))
    idx_whole = torch.LongTensor(range(nums[-1][2]))
    # print(idx_train)
    # print('nclass',labels.max().item() + 1)
    #
    model = GCN(nfeat=32,
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    t_total = time.time()
    if source_domain == target_domain:
        for epoch in range(args.epochs):
            train(epoch, model, optimizer, max_month, review_ground_truth, idx_map, rev_time, labels, \
                  features, nums, idx_train, idx_val, user_ground_truth, list_idx)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        torch.save(model.state_dict(), data_prefix + 'GCN_model_in_' + target_domain + '.pth')
    else:
        model.load_state_dict(torch.load(data_prefix + 'GCN_model_in_' + source_domain + '.pth'))

    model.load_state_dict(torch.load(data_prefix + 'GCN_model_in_' + source_domain + '.pth'))

    month1 = 0
    step = 3
    for month2 in range(12, max_month + step, step):
        adj =construct_adj_matrix(review_ground_truth, idx_map, labels, rev_time, month1, month2, 'month')
        # adj = adj_old + sp.eye(adj_old.shape[0])
        # adj = utils.normalize(adj + sp.eye(adj.shape[0]))
        # print('adj', adj)

        # adj_0=np.array(adj.todense(),copy=True)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        test(adj,model,features,nums,idx_test,labels,review_ground_truth,list_idx,user_ground_truth)

    print('training_domain:' + source_domain + ',test_domain:' + target_domain)

def train(epoch,model,optimizer,max_month,review_ground_truth,idx_map,rev_time,labels,\
          features,nums,idx_train,idx_val,user_ground_truth,list_idx):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    loss_train = 0
    acc_train = 0
    loss_val = 0
    acc_val = 0
    month1 = 0
    step = 6

    count=0

    for month2 in range(40, max_month + step, step):
        print('month2', month2)
        adj = construct_adj_matrix(review_ground_truth, idx_map, labels, rev_time, month1, month2, 'month')
        # adj = adj_old + sp.eye(adj_old.shape[0])
        # adj = utils.normalize(adj + sp.eye(adj.shape[0]))
        # print('adj', adj)

        # adj_0=np.array(adj.todense(),copy=True)
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        # print('adj', adj)

        output = model(features, adj, nums)
        loss = F.cross_entropy(output[idx_train], labels[idx_train])
        loss_train = loss + loss_train
        acc = accuracy(output[idx_train], labels[idx_train])
        acc_train = acc_train + acc
        count+=1

    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()

    model.eval()

    review_auc_zong=0
    user_auc_zong=0



    for month2 in range(40, max_month + step, step):
        adj = construct_adj_matrix(review_ground_truth, idx_map, labels, rev_time, month1, month2, 'month')
        # adj = adj_old + sp.eye(adj_old.shape[0])
        # adj = utils.normalize(adj + sp.eye(adj.shape[0]))
        # print('adj', adj)

        # adj_0=np.array(adj.todense(),copy=True)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        output = model(features, adj, nums)

        loss = F.cross_entropy(output[idx_val], labels[idx_val])
        loss_val = loss_val + loss
        acc = accuracy(output[idx_val], labels[idx_val])
        acc_val = acc_val + acc


        review_auc = auc_score(output, review_ground_truth, list_idx, idx_val, 'r')
        user_auc = auc_score(output, user_ground_truth, list_idx, idx_val, 'u')
        review_auc_zong+=review_auc
        user_auc_zong+=user_auc

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item() / count),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item() / count),
          'review_auc: {:.4f}'.format(review_auc_zong.item()/count),
          'user_auc: {:.4f}'.format(user_auc_zong.item()/count),
          'time: {:.4f}s'.format(time.time() - t))


def test(adj,model,features,nums,idx_test,labels,review_ground_truth,list_idx,user_ground_truth):
    output = model(features, adj, nums)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    review_auc = auc_score(output, review_ground_truth, list_idx, idx_test, 'r')
    user_auc = auc_score(output, user_ground_truth, list_idx, idx_test, 'u')
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "review_auc= {:.4f}".format(review_auc),
          "user_auc= {:.4f}".format(user_auc))




