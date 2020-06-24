"""Supervised GCN model class methods and skeleton"""

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

from gcn.core.model import Model

class SupervisedModel(Model):
    """Provides methodology for a supervised gcn model
    fit(): Runs model training procedure
    predict(): Predicts labels given the trained model
    score(): Scores the model (mmean accuracy)
    """

    def __init__(self, name):
        """Constructor"""
        Model.__init__(self, name)

        self.y_ = None
        self.accuracy = None

    def fit(self, trainA, trainY , trainX = None, valA = None, valX = None, valY = None):
        """Fits the model to the training data
        
        :param trainA: Graph adjacency matrix, array_like, shape (n_nodes, n_nodes)
        :param trainX: Node features, array_like, shape (n_nodes, n_features)
        :param trainY: Training labels, array_like, shape (n_nodes, n_classes)
        :param valA: Validation adjacency data, array_like, shape (N, n_nodes), (default = None)
        :param valX: Validation features, array_like, shape (N, n_features), (default = None)
        :param valY: Validation labels, array_like, shape (N, n_features), (default = None)        
        """
        
        n_nodes = trainA.shape[0]
        print('The number of nodes is: ', n_nodes)
        if self.feature_type == 'identity':
            n_features = n_nodes
            trainX = np.eye(n_features)
        n_classes = trainY.shape[1]
        feature_type = self.feature_type
        n_hidden = self.n_hidden
        act_func = self.act_func
        reg_type = self.reg_type
        reg_beta = self.reg_beta
        drop_rate = self.drop_rate
        
        
        #Build the model
        self.build_model(n_nodes, n_features, n_classes, feature_type, n_hidden,
                    act_func, reg_type, reg_beta, drop_rate)
        
        #Test the model
        self._predict(trainA, trainX)
        
        #Train the model
        # self._train_model(trainA, trainX, trainY, valA, valX, valY)
        
        return self

    def predict():
        return

    def score():
        return
