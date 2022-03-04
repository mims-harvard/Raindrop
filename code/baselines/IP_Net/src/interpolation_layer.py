"""
Adapted from: https://github.com/mlds-lab/interp-net
Works with Tensorflow 1
"""

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import keras
from keras import activations


class single_channel_interp(Layer):

    def __init__(self, ref_points, hours_look_ahead, **kwargs):
        self.ref_points = ref_points
        self.hours_look_ahead = hours_look_ahead  # in hours
        super(single_channel_interp, self).__init__(**kwargs)

    def build(self, input_shape):
        #input_shape [batch, features, time_stamp]
        self.time_stamp = input_shape[2]
        self.d_dim = input_shape[1] // 4
        self.activation = activations.get('sigmoid')
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.d_dim, ),
            initializer=keras.initializers.Constant(value=0.0),
            trainable=True)
        super(single_channel_interp, self).build(input_shape)

    def call(self, x, reconstruction=False):
        self.reconstruction = reconstruction
        x_t = x[:, :self.d_dim, :]
        d = x[:, 2*self.d_dim:3*self.d_dim, :]
        if reconstruction:
            output_dim = self.time_stamp
            m = x[:, 3*self.d_dim:, :]
            ref_t = K.tile(d[:, :, None, :], (1, 1, output_dim, 1))
        else:
            m = x[:, self.d_dim: 2*self.d_dim, :]
            ref_t = np.linspace(0, self.hours_look_ahead, self.ref_points)
            output_dim = self.ref_points
            ref_t.shape = (1, ref_t.shape[0])
        #x_t = x_t*m
        d = K.tile(d[:, :, :, None], (1, 1, 1, output_dim))
        mask = K.tile(m[:, :, :, None], (1, 1, 1, output_dim))
        x_t = K.tile(x_t[:, :, :, None], (1, 1, 1, output_dim))
        norm = (d - ref_t)*(d - ref_t)
        a = K.ones((self.d_dim, self.time_stamp, output_dim))
        pos_kernel = K.log(1 + K.exp(self.kernel))
        alpha = a*pos_kernel[:, np.newaxis, np.newaxis]
        w = K.logsumexp(-alpha*norm + K.log(mask), axis=2)
        w1 = K.tile(w[:, :, None, :], (1, 1, self.time_stamp, 1))
        w1 = K.exp(-alpha*norm + K.log(mask) - w1)
        y = K.sum(w1*x_t, axis=2)
        if reconstruction:
            rep1 = tf.concat([y, w], 1)
        else:
            w_t = K.logsumexp(-10.0*alpha*norm + K.log(mask),
                              axis=2)  # kappa = 10
            w_t = K.tile(w_t[:, :, None, :], (1, 1, self.time_stamp, 1))
            w_t = K.exp(-10.0*alpha*norm + K.log(mask) - w_t)
            y_trans = K.sum(w_t*x_t, axis=2)
            rep1 = tf.concat([y, w, y_trans], 1)
        return rep1

    def compute_output_shape(self, input_shape):
        if self.reconstruction:
            return (input_shape[0], 2*self.d_dim, self.time_stamp)
        return (input_shape[0], 3*self.d_dim, self.ref_points)


class cross_channel_interp(Layer):

    def __init__(self, **kwargs):
        super(cross_channel_interp, self).__init__(**kwargs)

    def build(self, input_shape):
        self.d_dim = input_shape[1] // 3
        self.activation = activations.get('sigmoid')
        self.cross_channel_interp = self.add_weight(
            name='cross_channel_interp',
            shape=(self.d_dim, self.d_dim),
            initializer=keras.initializers.Identity(gain=1.0),
            trainable=True)

        super(cross_channel_interp, self).build(input_shape)

    def call(self, x, reconstruction=False):
        self.reconstruction = reconstruction
        self.output_dim = K.int_shape(x)[-1]
        cross_channel_interp = self.cross_channel_interp
        y = x[:, :self.d_dim, :]
        w = x[:, self.d_dim:2*self.d_dim, :]
        intensity = K.exp(w)
        y = tf.transpose(y, perm=[0, 2, 1])
        w = tf.transpose(w, perm=[0, 2, 1])
        w2 = w
        w = K.tile(w[:, :, :, None], (1, 1, 1, self.d_dim))
        den = K.logsumexp(w, axis=2)
        w = K.exp(w2 - den)
        mean = K.mean(y, axis=1)
        mean = K.tile(mean[:, None, :], (1, self.output_dim, 1))
        w2 = K.dot(w*(y - mean), cross_channel_interp) + mean
        rep1 = tf.transpose(w2, perm=[0, 2, 1])
        if reconstruction is False:
            y_trans = x[:, 2*self.d_dim:3*self.d_dim, :]
            y_trans = y_trans - rep1  # subtracting smooth from transient part
            rep1 = tf.concat([rep1, intensity, y_trans], 1)
        return rep1

    def compute_output_shape(self, input_shape):
        if self.reconstruction:
            return (input_shape[0], self.d_dim, self.output_dim)
        return (input_shape[0], 3*self.d_dim, self.output_dim)
