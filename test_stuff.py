from wavenet import Wavenet
import tensorflow as tf
import time, os, math
import numpy as np
import tqdm as tqdm
from mu_law_ops import *
from dataset import *

if tf.__version__ == '1.14.0':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.logging.set_verbosity(tf.logging.ERROR)


WEIGHT_DECAY = 1e-10
def _encoder(x):

    k_init = 32

    b = tf.shape(x)[0]
    o = tf.zeros([b, k_init - 1, 1])
    x = tf.concat([o, x], 1)

    k_init = 32
    x = tf.layers.conv1d(
      inputs=x,
      filters=128,
      kernel_size=k_init,
      kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
      name='initial_filtering',
      kernel_initializer=tf.initializers.variance_scaling(
        scale=1.43,
        distribution='uniform'),
    )
    x = tf.nn.leaky_relu(x, 2e-2)

    x = tf.reverse(x, [1])  # paired op to enforce causality
    print('x:', x.shape)
    for i in range(6):
      conv = tf.layers.conv1d(
        inputs=x,
        filters=(i + 1) * 128,
        kernel_size=5,
        strides=2,
        padding='same',
        # activation=tf.nn.tanh,
        kernel_initializer=tf.initializers.variance_scaling(
          scale=1.15,
          distribution='uniform'),
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
      )
      gate = tf.layers.conv1d(
        inputs=x,
        filters=(i + 1) * 128,
        kernel_size=5,
        strides=2,
        padding='same',
        # activation=tf.nn.sigmoid,
        kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
        bias_initializer=tf.initializers.ones,
      )
      x = tf.nn.tanh(conv) * tf.nn.sigmoid(gate)
      print('x:', x.shape)
    x = tf.reverse(x, [1])  # paired op to enforce causality
    print('x:', x.shape)

    x = tf.layers.conv1d(
      inputs=x,
      filters=512,
      kernel_size=1,
      kernel_initializer=tf.initializers.variance_scaling(
          distribution='uniform'),
      kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
    )
    print('en:', x.shape)
    return x

x = tf.ones([1, 5120, 1])
x = _encoder(x)

