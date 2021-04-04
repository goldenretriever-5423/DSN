# -*-coding:utf-8-*-

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score
from collections import namedtuple

tf.compat.v1.disable_eager_execution()


class Graph(object):
    def __init__(self, args):
        self.params = []  # for computing regularization loss
        self._build_inputs(args)
        self._build_model(args)
        self._build_train(args)

    def _build_inputs(self, args):
        with tf.name_scope('input'):
            self.users_words = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, args.max_history_per_user, args.max_title_length, args.word_dim],
                name='clicked_words')
            self.clicked_nodes = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, args.max_history_per_user, args.node_dim], name='clicked_nodes')
            self.clicked_entities = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, args.max_history_per_user, args.max_entity_length, args.entity_dim],
                name='clicked_entities')
            self.news_words = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, args.max_title_length, args.word_dim], name='news_words')
            self.news_node = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, args.node_dim], name='news_node')
            self.news_entity = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, args.max_entity_length, args.entity_dim], name='news_entity')
            self.labels = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, args):

        # get users and news embeddings
        user_embeddings, news_embeddings = self._attention(args)
        # concat the embeddings
        # whether use node embeddings

        concat_embeddings = tf.concat([news_embeddings, user_embeddings], axis=1)

        dense1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu, name='dense1', \
                                       kernel_regularizer=tf.keras.regularizers.l2(args.l2_weight))(concat_embeddings)
        # dense1_dropout = tf.keras.layers.Dropout(0.2)(dense1)

        dense2 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu, name='dense2', \
                                       kernel_regularizer=tf.keras.regularizers.l2(args.l2_weight))(dense1)
        # dense2_dropout = tf.keras.layers.Dropout(0.5)(dense2)

        # dense3 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu, name='dense2', \
        #                                kernel_regularizer=tf.keras.regularizers.l2(args.l2_weight))(dense2)
        # dense3_dropout = tf.keras.layers.Dropout(0.2)(dense3)

        output = tf.keras.layers.Dense(units=1, name='predictions')(dense2)
        self.logits = tf.reduce_sum(output, axis=1)
        self.scores = tf.sigmoid(self.logits)

    def _attention(self, args):

        # reduce mean sen2vec  (title_length * word_embedding) -> (word_embedding dimension)
        # (batch_size, max_click_history, emb_dim) for users
        # (batch_size, emb_dim) for news
        # clicked_words = tf.reduce_mean(self.users_words, axis=2)
        # news_embeddings = tf.reduce_mean(self.news_words, axis=1)

        # expand dimension of nodes
        clicked_nodes = tf.expand_dims(self.clicked_nodes, 2)
        news_node = tf.expand_dims(self.news_node, 1)

        # copy the node embedding
        clicked_nodes = tf.tile(clicked_nodes, [1, 1, args.max_title_length, 1])
        news_node = tf.tile(news_node, [1, args.max_title_length, 1])

        if args.use_knowledge:
            # expand dimension of entity
            clicked_entities = tf.tile(self.clicked_entities, [1, 1, 2, 1])
            news_entity = tf.tile(self.news_entity, [1, 2, 1])
        if args.transform_entity:
            clicked_entities = tf.keras.layers.Dense(units=args.word_dim, activation=tf.nn.tanh,\
                                                     name='transformed_clikied_entities', \
                                                     kernel_regularizer=tf.keras.regularizers.l2(args.l2_weight))(clicked_entities)
            news_entity = tf.keras.layers.Dense(units=args.word_dim, activation=tf.nn.tanh,\
                                                name='transformed_news_entity', \
                                                kernel_regularizer=tf.keras.regularizers.l2(args.l2_weight))(news_entity)
        if args.transform_node:
            clicked_nodes = tf.keras.layers.Dense(units=args.word_dim, activation=tf.nn.tanh, \
                                                     name='transformed_clikied_nodes', \
                                                     kernel_regularizer=tf.keras.regularizers.l2(args.l2_weight))(clicked_nodes)
            news_node = tf.keras.layers.Dense(units=args.word_dim, activation=tf.nn.tanh,\
                                                name='transformed_news_node', \
                                                kernel_regularizer=tf.keras.regularizers.l2(args.l2_weight))(news_node)

        # reshape the input data
        clicked_words = tf.reshape(self.users_words, shape=[-1, args.max_title_length, args.word_dim])
        clicked_nodes = tf.reshape(clicked_nodes, shape=[-1, args.max_title_length, args.word_dim])
        if args.use_knowledge:
            clicked_entities = tf.reshape(clicked_entities, shape=[-1, args.max_title_length, args.word_dim])

        with tf.compat.v1.variable_scope('kcnn', reuse=tf.compat.v1.AUTO_REUSE):
            # (batch_size * max_click_history, title_embedding_length)
            # title_embedding_length = n_filters_for_each_size * n_filter_sizes
            clicked_embeddings = self._kcnn(clicked_words, clicked_nodes, args)
            news_embeddings = self._kcnn(self.news_words, news_node, args)

        # (batch_size, max_click_history, title_embedding_length)
        clicked_embeddings = tf.reshape( \
            clicked_embeddings, shape=[-1, args.max_history_per_user, args.n_filters * len(args.filter_sizes)])

        # (batch_size, 1, title_embedding_length)
        news_embeddings_expanded = tf.expand_dims(news_embeddings, 1)

        # (batch_size, max_click_history)
        attention_weights = tf.reduce_sum(clicked_embeddings * news_embeddings_expanded, axis=-1)

        # (batch_size, max_click_history)
        attention_weights = tf.compat.v1.nn.softmax(attention_weights, dim=-1)

        # (batch_size, max_click_history, 1)
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)

        # (batch_size, title_embedding_length)
        user_embeddings = tf.reduce_sum(clicked_embeddings * attention_weights_expanded, axis=1)

        return user_embeddings, news_embeddings

    def _kcnn(self, words, nodes, args):
        # (batch_size * max_click_history, word_dim) for users words, (batch_size * max_click_history, node_dim) for users_nodes
        # (batch_size, word_dim) for news words, (batch_size, word_dim) for news node

        # (batch_size * max_click_history, max_title_length, full_dim) for users
        # (batch_size, max_title_length, full_dim) for news
        if args.use_knowledge:
            concat_input = tf.concat([words, nodes], axis=-1)
            full_dim = args.word_dim * 2 + args.node_dim
        else:
            concat_input = tf.concat([words, nodes], axis=-1)
            full_dim = args.word_dim *2

        # (batch_size * max_click_history, max_title_length, full_dim, 1) for users
        # (batch_size, max_title_length, full_dim, 1) for news
        concat_input = tf.expand_dims(concat_input, -1)

        outputs = []
        for filter_size in args.filter_sizes:
            filter_shape = [filter_size, full_dim, 1, args.n_filters]
            w = tf.compat.v1.get_variable(name='w_' + str(filter_size), shape=filter_shape, dtype=tf.float32)
            b = tf.compat.v1.get_variable(name='b_' + str(filter_size), shape=[args.n_filters], dtype=tf.float32)
            if w not in self.params:
                self.params.append(w)

            # (batch_size * max_click_history, max_title_length - filter_size + 1, 1, n_filters_for_each_size) for users
            # (batch_size, max_title_length - filter_size + 1, 1, n_filters_for_each_size) for news
            conv = tf.nn.conv2d(concat_input, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
            relu = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

            # (batch_size * max_click_history, 1, 1, n_filters_for_each_size) for users
            # (batch_size, 1, 1, n_filters_for_each_size) for news
            pool = tf.nn.max_pool(relu, ksize=[1, args.max_title_length - filter_size + 1, 1, 1],
                                  strides=[1, 1, 1, 1], padding='VALID', name='pool')
            outputs.append(pool)

        # (batch_size * max_click_history, 1, 1, n_filters_for_each_size * n_filter_sizes) for users
        # (batch_size, 1, 1, n_filters_for_each_size * n_filter_sizes) for news
        output = tf.concat(outputs, axis=-1)

        # (batch_size * max_click_history, n_filters_for_each_size * n_filter_sizes) for users
        # (batch_size, n_filters_for_each_size * n_filter_sizes) for news
        output = tf.reshape(output, [-1, args.n_filters * len(args.filter_sizes)])

        return output


    def _build_train(self, args):
        with tf.name_scope('train'):
            self.base_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
            self.l2_loss = tf.Variable(tf.constant(0., dtype=tf.float32), trainable=False)
            for param in self.params:
                self.l2_loss = tf.add(self.l2_loss, args.l2_weight * tf.nn.l2_loss(param))

            self.loss = self.base_loss + self.l2_loss
            self.optimizer = tf.compat.v1.train.AdamOptimizer(args.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run(self.optimizer, feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores4F1 = [int(item > 0.5) for item in scores]
        F1 = f1_score(y_true=labels, y_pred=scores4F1)
        return auc, F1

