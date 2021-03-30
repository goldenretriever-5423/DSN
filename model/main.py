print("learning rate: ", args.lr)
print("MLP structure: ",(256,256))
print('dropout :',())
print("batch size: ", args.batch_size)
# print("use social node: ", args.use_nodes)
print('num of click history: ', args.max_history_per_user)
print('normalized :', 'word only')
print('n_filter :', args.n_filters)


train_data, test_data = load_data(args)
train(args, train_data, test_data)