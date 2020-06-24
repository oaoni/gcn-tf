"""Provide utility for loading sequence data"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os
import networkx as nx

def load_graph(dataset):
    
    if dataset == 'default':
        graph = _load_karate_club_graph()
        
    return graph

def _load_karate_club_graph():
    
    graph = {}
    node_attr = []
    G = nx.karate_club_graph()
    A = nx.convert_matrix.to_numpy_matrix(G)
    
    for node in G.nodes(data=True):
        node_attr += [node[1]['club']]
    
    # lb = preprocessing.LabelBinarizer()
    # labels = lb.fit_transform(node_attr)

    enc = OneHotEncoder(categories='auto')
    labels = enc.fit_transform(np.array(node_attr).reshape(-1,1)).toarray()
    
    graph['A'] = A
    graph['Y'] = labels

    return graph
