"""Graph Convolutional Network based classification using Tensorflow """

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tensorflow.python.keras.initializers import Identity, Zeros
from tensorflow.python.keras.layers import Dropout, Layer, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

#from dgmu.core import Trainer, Layers, Score, Loss
from gcn.core import GraphConvolution
from gcn.core import SupervisedModel



class GCN(SupervisedModel):
    """GCN using Tensorflow 2"""

    def __init__(self, name = 'gcn', epochs = 100, batch_size = 100,
                 cost_func = 'softmax_cross_entropy', lr = 0.01, drop_rate=0,
                 n_hidden = 50, input_dr = 0, optimizer = 'adam',
                 reg_type = 'l2', reg_beta = 0, save_step = 100,        
                 convolution='spectral', feature_type='identity', act_func = 'relu'):
        """Constructor"""
        SupervisedModel.__init__(self, name)

        self.epochs = epochs
        self.batch_size = batch_size
        self.cost_func = cost_func
        self.lr = lr
        self.n_hidden = n_hidden
        self.input_dr = input_dr
        self.optimizer = optimizer
        self.reg_type = reg_type
        self.reg_beta = np.float32(reg_beta)
        self.save_step = save_step
        self.convolution = convolution
        self.feature_type = feature_type
        self.n_hidden = n_hidden
        self.act_func = act_func
        self.drop_rate = drop_rate

        # self.loss = Loss(cost_func)
        # self.trainer = Trainer(optimizer, lr)
        
    def build_model(self, n_nodes, n_features, n_classes, feature_type, n_hidden,
                    act_func, reg_type, reg_beta, drop_rate):
        
        """Create the computational graph
        :param n_nodes: number of nodes in graph
        :param n_features: number of features
        :param n_classes: number of classes
        :param feature_type: type of feature
        :return:
        """
        
        #Select convolution aggregator operator
        
        #Create placeholders
        A = Input(shape=(n_nodes,), dtype=tf.float32)

        # if feature_type == 'identity':
        X = Input(shape=(n_features,n_features), dtype=tf.float32, name='node_feat')
        #Create graph
        # self.y = GraphConvolution(n_hidden, act_func, reg_type, reg_beta, drop_rate)([self.A,self.X])
        Y = GraphConvolution(units=n_hidden)([A,X])
        # self.y = tf.keras.layers.Dense(n_hidden, input_shape=(None,n_nodes))(self.A)
        
        self.model = Model(inputs=[A,X], outputs=Y)
            
            
    def _train_model(self, trainA, trainX, trainY, valA, valX, valY):
        """Train the model
        :param trainA: Graph adjacency matrix, array_like, shape (n_nodes, n_nodes)
        :param trainX: Node features, array_like, shape (n_nodes, n_features)
        :param trainY: Training labels, array_like, shape (n_nodes, n_classes)
        :param valA: Validation adjacency data, array_like, shape (N, n_nodes), (default = None)
        :param valX: Validation features, array_like, shape (N, n_features), (default = None)
        :param valY: Validation labels, array_like, shape (N, n_features), (default = None)        
        :return self: trained model instance
        """
        loss_object = SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam()
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
        self.model.compile(optimizer=Adam(0.01), loss=loss_object, metrics=['accuracy'])
        
        
    def _predict(self, trainA, trainX):
        
        print(trainX.shape)
        predictions = self.model([trainA, trainX]).numpy()
        
        return predictions








