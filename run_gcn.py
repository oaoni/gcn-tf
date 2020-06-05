from absl import flags
from absl import logging
from absl import app
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import sys
import networkx as nx

from gcn.models import gcn
from gcn.utils import datasets

#Disables AVX compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define flags
FLAGS = flags.FLAGS


#Global configuration
flags.DEFINE_string('dataset', 'default',help='The data set to use. ["default"]')
flags.DEFINE_string('name', 'gmu', 'Model name.')

#GCN specific configurations
flags.DEFINE_integer('hidden_dim', 200, 'Number of hidden units.')
flags.DEFINE_integer('save_step', 10, 'Number of iterations to save summaries to filewriter.')




def main(argv):

    #Load desired dataset
    if FLAGS.dataset == 'default':

        G = datasets.load_karate_club_graph()

        # print("Node Degree")
        # for v in G:
        #     print('%s %s' % (v, G.degree(v)))
        #
        # nx.draw_circular(G, with_labels=True)
        # plt.show()

    print(FLAGS.dataset)

    #Run desired model
    #Create the model object
    GCN = gcn.GCN(name=FLAGS.name, hidden_dim=FLAGS.hidden_dim,
    save_step = FLAGS.save_step)

    print(GCN)
    print("the gift that keeps giving like Babushka")



if __name__ == "__main__":

    app.run(main)
