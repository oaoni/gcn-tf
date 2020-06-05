"""Basic model class methods and skeleton"""

from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

#from gcn.core import Borg, Config

class Model(object):
    """Generic abstract model in tensorflow"""

    def __init__(self, name):
        """Constructor
        :param name: string, name of the model and also the model filename
        """

        self.name = name
        self.graph = None
        self.side = None
        self.dropout_rate = None
        self.layer_nodes = []
        self.train_step = None
        self.cost = None

        #Tensorflow objects
        self.tf_graph = None
        self.tf_session = None
        self.tf_saver = None
        self.tf_merged_summaries = None
        self.tf_train_writer = None
        self.tf_test_writer = None
