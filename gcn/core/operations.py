"""Neural network layer operations"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.regularizers import l2,l1


# class GraphConvolution(layers.Layer):
#     """Performs graph convolution
#     :param 
    
#     """
# # n_hidden, act_func, reg_type, reg_beta, drop_rate
#     def __init__(self, n_hidden, act_func, reg_type, reg_beta, drop_rate):
#       super(GraphConvolution, self).__init__()
#       self.n_hidden = n_hidden
#       self.act_func = act_func
#       self.reg_beta = reg_beta
#       self.reg_type = reg_type
#       self.drop_rate = drop_rate
#       if reg_type == 'l2':
#           self.reg = l2(self.reg_beta)
#       if reg_type == 'l1':
#           self.reg = l1(self.reg_beta)
    
#     def build(self, input_shapes):
#       self.W = self.add_weight(shape=(input_shapes[0][-1], self.n_hidden),
#                                initializer='random_normal',
#                                regularizer=self.reg,
#                                trainable=True)
#       self.b = self.add_weight(shape=(self.n_hidden,),
#                                initializer='random_normal',
#                                trainable=True)
    
#     def call(self, inputs, aggOp="test"):
        
#         A, X = inputs
#         # #sparse_tensor_dense_matmul
#         # if aggOp == 'spectral':
#         #     D = tf.math.sqrt(tf.math.reciprocal(tf.math.reduce_sum(A,1)))
#         #     D = tf.linalg.tensor_diag(D)
#         #     DAD = tf.matmul(D,tf.matmul(A,D))
#         #     AX = tf.matmul(DAD,X)
            
#         # if aggOp == 'mean':
#         #     D = tf.math.reciprocal(tf.math.reduce_sum(A,1))
#         #     D = tf.linalg.tensor_diag(D)
#         #     AX = tf.matmul(D,tf.matmul(A,X))
            
#         # if aggOp == 'sum':
#         #     AX = tf.matmul(A,X)
            
#         # if aggOp == 'test':
#         #     AX = tf.matmul(A,X)
        
#         # AX = tf.matmul(A,X)
            
#         # return tf.matmul(AX, self.W) + self.b
#         return A

class GraphConvolution(layers.Layer):
    def __init__(self, units=32):
        super(GraphConvolution, self).__init__()
        self.units = units

    def build(self, input_shape):
        print(input_shape[0][-1], self.units)
        self.w = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        A = inputs[0]
        X = inputs[1]
        # return tf.matmul(A, self.w) + self.b
        return X
    
