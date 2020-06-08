"""Neural network layer operations"""

import tensorflow as tf
from tensorflow.keras import layers


class GraphConvolution(layers.Layer):
    """Performs graph convolution
    :param 
    
    """

    def __init__(self, n_features=32):
      super(GraphConvolution, self).__init__()
      self.n_features = n_features
    
    def build(self, input_shape):
      self.w = self.add_weight(shape=(input_shape[-1], self.n_features),
                               initializer='random_normal',
                               trainable=True)
      self.b = self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)
    
    def call(self, inputs, aggOp="spectral"):
        
        if aggOp == 'spectral':
            D = None
            A = tf.matul(tf.matmul(D,A),D)
        
        
        return tf.matmul(X, self.W) + self.b