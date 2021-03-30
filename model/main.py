from config import args
from train import train
from utils.load_data import load_data


print("learning rate: ", args.lr)
print("batch size: ", args.batch_size)
print('num of click history: ', args.max_history_per_user)
print('n_filter :', args.n_filters)


train_data, test_data = load_data(args)
train(args, train_data, test_data)