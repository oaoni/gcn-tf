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

    def __init__(self, name, n_features):
        """Constructor"""
        Model.__init__(self, name)

        self.n_features = n_features

        self.y_ = None
        self.accuracy = None

    def fit():
        return

    def predict():
        return

    def score():
        return
