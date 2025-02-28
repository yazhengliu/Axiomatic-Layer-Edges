import sys
from train import train_pheme,train_weibo,train_yelp,train_graph,train_link
import argparse

def main(args):
    # 获取传入的参数
    dataset=args.dataset
    if dataset == "pheme":
        train_pheme.train_all(args)
    elif dataset == "weibo":
        train_weibo.train_all(args)
    elif dataset=='Chi' or dataset=='NYC':
        train_yelp.train_all(args)
    elif dataset=='mutag' or dataset=='REDDIT-BINARY' or dataset=='IMDB-BINARY' or dataset=='ClinTox':
        train_graph.train_all(args)
    elif dataset=='bitcoinotc' or dataset=='bitcoinalpha' or dataset=='UCI':
        train_link.train_all(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='UCI')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--glove_embedding', type=float, default=None,
                      )
    parser.add_argument('--num_layers', type=int, default=2,
                        )

    parser.add_argument('--train_ratio', type=float, default=0.5)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    args = parser.parse_args()
    main(args)
