import random
import sys
import torch
import argparse
from explain.rumor_explain import explain_rumor
from explain.yelp_explain import explain_yelp
from explain.graph_explain import explain_graph
from explain.link_explain import explain_link
def main(args):
    # 获取传入的参数
    dataset=args.dataset
    if dataset == "pheme" or dataset=='weibo':
        explain_rumor(args)
    elif dataset=='Chi' or dataset=='NYC':
        explain_yelp(args)
    elif dataset=='mutag' or dataset=='clintox' or dataset=='REDDIT-BINARY' or  dataset=='IMDB-BINARY':
        explain_graph(args)
    elif dataset=='bitcoinalpha' or dataset=='bitcoinotc' or dataset=='UCI':
        explain_link(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='UCI')
    parser.add_argument('--layernumbers', type=str, default=2)
    args = parser.parse_args()
    main(args)
