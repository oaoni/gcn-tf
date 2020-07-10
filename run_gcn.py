from absl import flags
# from absl import logging
# from absl import app
# import numpy as np
# import pandas as pd
# import tensorflow as tf
import os
import matplotlib.pyplot as plt
import sys
import networkx as nx

from gcn.models import gcn
from gcn.utils import datasets
import importlib

#Disables AVX compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define flags
FLAGS = flags.FLAGS


#Global configuration
flags.DEFINE_string('dataset', 'default',help='The data set to use. ["default"]')
flags.DEFINE_string('name', 'gmu', 'Model name.')

#GCN specific configurations
flags.DEFINE_integer('n_hidden', 40, 'Number of hidden units.')
flags.DEFINE_integer('save_step', 10, 'Number of iterations to save summaries to filewriter.')



if __name__ == "__main__":
    

    flags.FLAGS(sys.argv)
    
    graph = datasets.load_graph(FLAGS.dataset)

    #Run desired model
    #Create the model object
    # name = 'gcn', epochs = 100, batch_size = 100,
    #              cost_func = 'softmax_cross_entropy', lr = 0.01, drop_rate=0,
    #              n_hidden = 500, input_dr = 0, optimizer = 'adam',
    #              reg_type = 'l2', reg_beta = 0, n_features = 500, save_step = 100,
    #              convolution='spectral', feature_type='identity',n_nodes=10)
    GCN = gcn.GCN(name=FLAGS.name, n_hidden=FLAGS.n_hidden, 
    save_step = FLAGS.save_step)
    
    #Fit the model
    pred = GCN.fit(graph['A'][0,:].reshape(1,-1), graph['Y'][0,:].reshape(1,-1))
    
   
    print("le cadeau qui continue de donner comme Babouchka")
