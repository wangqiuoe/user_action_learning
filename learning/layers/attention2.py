#encoding:utf-8
# @author April Wang
# @date 2019-02-15

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
import numpy as np
class Attention(Layer):
    """
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(Attention())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 **kwargs):

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        super(Attention, self).__init__(**kwargs)
        

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.uw = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    def call(self, h):
        #h: shape[batch_size, timestep, feature]
        u = K.dot(h, self.W) + self.b
        u = K.tanh(u)
        uw_exp = K.expand_dims(self.uw, axis=-1)
        a = K.squeeze(K.dot(u, uw_exp), axis=-1)
        a = K.softmax(a) #attention weight on timestep
        #attention = a
        #a = K.expand_dims(a, axis=-1)
        #print '4 a.shape:', a.shape
        #h_after_a = h * a
        #h_after_a = K.sum(h_after_a, axis=1)
        #print '5 h_after_a.shape:', h_after_a.shape
        return a

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

