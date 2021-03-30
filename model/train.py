#-*-coding:utf-8-*-

from graph_cnn import Graph
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

# ops.reset_default_graph()


# def get_feed_dict(model, data, start, end):
#     feed_dict = {model.users_words: data.users_words[start:end],
#                  model.users_nodes: data.users_nodes[start:end],
#                  model.news_words: data.news_words[start:end],
#                  model.labels: data.labels[start:end]}
#     return feed_dict

def get_feed_dict(model, args, data, start, end):
    if args.use_knowledge:
        feed_dict = {model.users_words: data.clicked_words[start:end],
                    model.clicked_nodes: data.clicked_entities[start:end],
                    model.news_node: data.news_entity[start:end],
                    model.news_words: data.news_words[start:end],
                    model.labels: data.labels[start:end]}
    else:
        feed_dict = {model.users_words: data.clicked_words[start:end],
                     model.clicked_nodes: data.clicked_nodes[start:end],
                     model.news_node: data.news_node[start:end],
                     model.news_words: data.news_words[start:end],
                     model.labels: data.labels[start:end]}
    return feed_dict




def train(args, train_data, test_data):
    ops.reset_default_graph()
    model = Graph(args)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        for step in range(args.n_epochs):
            # training
            start_list = list(range(0, train_data.size, args.batch_size))
            np.random.shuffle(start_list)
            for start in start_list:
                end = start + args.batch_size
                model.train(sess, get_feed_dict(model, args, train_data, start, end))

            # evaluation
            train_auc, train_F1 = model.eval(sess, get_feed_dict(model, args, train_data, 0, train_data.size))
            test_auc, test_F1 = model.eval(sess, get_feed_dict(model, args, test_data, 0, test_data.size))
            print('epoch %d    train_auc: %.4f     train_F1: %.4f   test_auc: %.4f    test_F1: %.4f' \
                  % (step, train_auc, train_F1, test_auc, test_F1))
