import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--use_sample', type=bool, default=False, help='whether to use sample data')
parser.add_argument('--sample_num', type=int, default=7000, help='sampling number')
parser.add_argument('--use_nodes', type=bool, default=True, help='whether to use social node embeddings')
parser.add_argument('--max_title_length', type=int, default=10, help='max length of news titles')
parser.add_argument('--max_history_per_user', type=int, default=25, help='number of sampled history per user')

# transformation
parser.add_argument('--node_dim', type=int, default=50,
                    help='dimension of social network node embeddings')
parser.add_argument('--word_dim', type=int, default=300,
                    help='dimension of word embeddings')
parser.add_argument('--transform_word', type=bool, default=True, help='whether to transform word embeddings')
parser.add_argument('--transform_node', type=bool, default=True, help='for alignment')
parser.add_argument('--node_dim_set', type=int, default=100,
                    help='dimension of transformed social network node embeddings')
parser.add_argument('--word_dim_set', type=int, default=100,
                    help='dimension of transformed word embeddings')

# graph
parser.add_argument('--batch_size', type=int, default=128, help='number of samples in one batch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--l2_weight', type=float, default=0.001, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')



# for Kim CNN paras
parser.add_argument('--n_filters', type=int, default=128, help='number of filters for each size in Kim CNN')
parser.add_argument('--filter_sizes', type=int, default=[1, 2], nargs='+',
                    help='list of filter sizes')

# for knowledge test
parser.add_argument('--use_knowledge', type=bool, default=False, help='whether to use knowledge embedding')
parser.add_argument('--entity_dim', type=int, default=512,
                    help='dimension of entity embeddings')
parser.add_argument('--transform_entity', type=bool, default=False, help='for alignment')
parser.add_argument('--max_entity_length', type=int, default=5, help='max #entity per title')


args = parser.parse_args()