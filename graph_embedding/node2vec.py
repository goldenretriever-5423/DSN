# -*- coding:utf-8 -*-

"""



Author:

    Weichen Shen,wcshen1994@163.com



Reference:

    [1] Grover A, Leskovec J. node2vec: Scalable feature learning for networks[C]//Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2016: 855-864.(https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf)



"""

from gensim.models import Word2Vec
import pandas as pd

from .walker import RandomWalker

import numpy as np

from ge.classify import read_node_label, Classifier
from ge import Node2Vec
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

import pickle


class Node2Vec:

    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1, use_rejection_sampling=0):

        self.graph = graph
        self._embeddings = {}
        self.walker = RandomWalker(
            graph, p=p, q=q, use_rejection_sampling=use_rejection_sampling)

        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()

        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=100, window_size=5, workers=3, iter=5, **kwargs):  # orig 100

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model

        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}

        ## modification
        # self.w2v_model.init_sims(replace=True)
        ## modification

        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings


def save_pkl(obj, pkl_name):
    with open(pkl_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    G=nx.read_edgelist('./weighted_edgelist.txt',
        # '../data/wiki/Wiki_edgelist.txt',
                         create_using = nx.DiGraph(), nodetype = None, data = [('weight', float)])
    model = Node2Vec(G, walk_length=10, num_walks=80,
                     p=0.25, q=4, workers=1, use_rejection_sampling=0)
    model.train(window_size = 5, iter = 3)
    embeddings=model.get_embeddings()

    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)

    # save embeddings to txt
    # path = r'/home/yuting/PycharmProjects/data_preprocessing/data_stat/weighted graph/sample_network_embedding.txt'
    save_pkl(embeddings,'node_embedding_100')
