#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
import pickle
from collections import namedtuple

# for weibo

Data1 = namedtuple('Data', ['size', 'clicked_words', 'clicked_nodes', 'news_words', 'news_node', 'labels'])
Data2 = namedtuple('Data', ['size', 'clicked_words', 'clicked_entities', 'news_words', 'news_entity', 'labels'])
features = ['news','history_titles','node_emb']

# #for twitter
# Data = namedtuple('Data', ['size', 'users_words', 'clicked_nodes', 'news_words', 'news_node', 'labels'])
# features = ['news','history_titles','node_emb']

# formed_data_history25_neg15_with_entity.pkl

# TODO the fraction 0.8
def load_data(args):
    global features
    df = load_pkl('../data/weibo_formed_data_1m_history25_neg15_influence_emb.pkl')
    df = df.sample(frac=1, random_state=3)
    df_train = df[0:100]
    df_test = df[100:120]

    train_data = transform(df_train,args)
    test_data = transform(df_test,args)

    return train_data, test_data

def save_pkl(obj, pkl_name):
    with open(pkl_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(pkl_name):
    with open(pkl_name, 'rb') as f:
        return pickle.load(f)

def transform(df,args):
    if args.use_knowledge:
        data = Data2(size=df.shape[0],
                    clicked_words=np.array(df['history_news'].tolist()),
                    clicked_entities=np.array(df['history_entities'].tolist()),
                    news_words=np.array(df['news'].tolist()),
                    news_entity=np.array(df['entities'].tolist()),
                    labels=np.array(df['labels']))
    else:
        data = Data1(size=df.shape[0],
                    clicked_words=np.array(df['history_news'].tolist()),
                    clicked_nodes=np.array(df['history_nodes'].tolist()),
                    news_words=np.array(df['news_emb'].tolist()),
                    news_node=np.array(df['orig_node'].tolist()),
                    labels=np.array(df['labels']))
    return data





if __name__=="__main__":

    path = '../data/news/users_titles_embedding.pkl'
    titles = load_pkl(path)
    print(len(titles))

