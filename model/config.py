import argparse
from train import train
from load_data import load_data
from collections import namedtuple
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='../data/news/users_titles_words.txt', help='path to the training file')
parser.add_argument('--test_file', type=str, default='../data/news/users_titles_words.txt', help='path to the test file')
parser.add_argument('--use_sample', type=bool, default=True, help='whether to use sample data')
parser.add_argument('--use_nodes', type=bool, default=False, help='whether to use social node embeddings')
parser.add_argument('--transform_node', type=bool, default=True, help='whether to transform entity embeddings')
parser.add_argument('--transform_entity', type=bool, default=False, help='whether to transform entity embeddings')
parser.add_argument('--max_title_length', type=int, default=10, help='the max length of news titles')
parser.add_argument('--max_entity_length', type=int, default=5, help='the max length of entity per title')
parser.add_argument('--max_history_per_user', type=int, default=25, help='number of sampled history per user')

parser.add_argument('--batch_size', type=int, default=128, help='number of samples in one batch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--l2_weight', type=float, default=0.001, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--node_dim', type=int, default=50,
                    help='dimension of social network node embeddings')
parser.add_argument('--word_dim', type=int, default=300,
                    help='dimension of word embeddings')
parser.add_argument('--entity_dim', type=int, default=512,
                    help='dimension of entity embeddings')
# # for CNN embedding
parser.add_argument('--n_filters', type=int, default=128, help='number of filters for each size in KCNN')
parser.add_argument('--filter_sizes', type=int, default=[1, 2], nargs='+',
                    help='list of filter sizes, e.g., --filter_sizes 2 3')
parser.add_argument('--use_knowledge', type=bool, default=False, help='whether to use knowledge embedding')
