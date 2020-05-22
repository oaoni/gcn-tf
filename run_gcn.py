import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from absl import app
from absl import flags
import sys
import networkx as nx



from gcn.utils import datasets

#Disables AVX compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define flags
FLAGS = flags.FLAGS


#Global configuration
flags.DEFINE_string('dataset', 'default', 'The data set to use. ["default"]')

FLAGS(sys.argv)

if __name__ == "__main__":

    if FLAGS.dataset == 'default':

        G = datasets.load_karate_club_graph()

        # print("Node Degree")
        # for v in G:
        #     print('%s %s' % (v, G.degree(v)))
        #
        # nx.draw_circular(G, with_labels=True)
        # plt.show()

    print("sfsg")
