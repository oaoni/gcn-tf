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

def load_karate_club_graph():

    G = nx.karate_club_graph()

    return G
