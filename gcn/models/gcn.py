"""Graph Convolutional Network based classification using Tensorflow """

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm

#from dgmu.core import Trainer, Layers, Score, Loss
from gcn.core import SupervisedModel

class GCN(SupervisedModel):
    """Biomodal GMU using Tensorflow"""

    def __init__(self, name = 'gcn', epochs = 100, batch_size = 100,
                 cost_func = 'softmax_cross_entropy', lr = 0.01,
                 hidden_dim = 500, input_dr = 0, optimizer = 'adam',
                 reg_type = 'l2', beta = 0, n_features = 500, save_step = 100):
        """Constructor"""
        SupervisedModel.__init__(self, name, n_features)

        self.epochs = epochs
        self.batch_size = batch_size
        self.cost_func = cost_func
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.input_dr = input_dr
        self.optimizer = optimizer
        self.reg_type = reg_type
        self.beta = np.float32(beta)
        self.save_step = save_step

        # self.loss = Loss(cost_func)
        # self.trainer = Trainer(optimizer, lr)
