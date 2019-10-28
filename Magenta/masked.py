# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A library of functions that help with causal masking."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def shift_right(x):
  shape = x.get_shape().as_list()
  x_padded = tf.pad(x, [[0, 0], [1, 0], [0, 0]])
  x_sliced = x_padded[:, :-1, :]
  x_sliced.set_shape(shape)
  return x_sliced

def mu_law(x, quantization_channels=256):
  mu = tf.cast(quantization_channels - 1, tf.float32)
  x = tf.clip_by_value(x, -1., 1.)
  y = tf.sign(x) * tf.math.log1p(mu * tf.abs(x)) / tf.math.log1p(mu)
  return y

def mul_or_none(a, b):
  if a is None or b is None:
    return None
  return a * b

def time_to_batch(x, block_size):
  shape1 = x.get_shape().as_list()
  shape = tf.shape(x)
  y = tf.reshape(x, [shape[0], shape[1] // block_size, block_size, shape[2]])
  y = tf.transpose(y, [0, 2, 1, 3])
  y = tf.reshape(y, [shape[0] * block_size, shape[1] // block_size, shape[2]])
  y.set_shape([
      mul_or_none(shape1[0], block_size), mul_or_none(shape1[1], 1. / block_size),
      shape1[2]])
  return y

def batch_to_time(x, block_size):
  shape1 = x.get_shape().as_list()
  shape = tf.shape(x)
  y = tf.reshape(x, [shape[0] // block_size, block_size, shape[1], shape[2]])
  y = tf.transpose(y, [0, 2, 1, 3])
  y = tf.reshape(y, [shape[0] // block_size, shape[1] * block_size, shape[2]])
  y.set_shape([mul_or_none(shape1[0], 1. / block_size),
      mul_or_none(shape1[1], block_size),
      shape1[2]])
  return y

def conv1d(x,
           num_filters,
           filter_length,
           name,
           hist=False,
           dilation=1,
           causal=True,
           kernel_initializer=tf.uniform_unit_scaling_initializer(1.0),
           biases_initializer=tf.constant_initializer(1.0),
           regularizer=None,
           is_training=True):
  batch_size, length, num_input_channels = x.get_shape().as_list()

  kernel_shape = [1, filter_length, num_input_channels, num_filters]
  strides = [1, 1, 1, 1]
  biases_shape = [num_filters]
  padding = 'VALID' if causal else 'SAME'

  with tf.variable_scope(name):
    weights = tf.get_variable(
        'W', shape=kernel_shape, initializer=kernel_initializer,
        regularizer=regularizer, trainable=is_training)
    biases = tf.get_variable(
        'biases', shape=biases_shape, initializer=biases_initializer,
        regularizer=regularizer, trainable=is_training)
    if hist:
        tf.summary.histogram(name + '_kernel', weights)
        tf.summary.histogram(name + '_bias', biases)

  x_ttb = time_to_batch(x, dilation)
  if filter_length > 1 and causal:
    x_ttb = tf.pad(x_ttb, [[0, 0], [filter_length - 1, 0], [0, 0]])

  x_ttb_shape = x_ttb.get_shape().as_list()
  x_4d = tf.expand_dims(x_ttb, 1)
  y = tf.nn.conv2d(x_4d, weights, strides, padding=padding)
  y = tf.nn.bias_add(y, biases)
  y_shape = y.get_shape().as_list()
  y = tf.squeeze(y, 1)
  y = batch_to_time(y, dilation)
  y.set_shape([batch_size, length, num_filters])
  return y

def pool1d(x, window_length, name, mode='avg', stride=None):
  """1D pooling function that supports multiple different modes.

  Args:
    x: The [mb, time, channels] float tensor that we are going to pool over.
    window_length: The amount of samples we pool over.
    name: The name of the scope for the variables.
    mode: The type of pooling, either avg or max.
    stride: The stride length.

  Returns:
    pooled: The [mb, time // stride, channels] float tensor result of pooling.
  """
  if mode == 'avg':
    pool_fn = tf.nn.avg_pool
  elif mode == 'max':
    pool_fn = tf.nn.max_pool

  stride = stride or window_length

  window_shape = [1, 1, window_length, 1]
  strides = [1, 1, stride, 1]
  x_4d = tf.expand_dims(x, axis=1)
  pooled = pool_fn(x_4d, window_shape, strides, padding='SAME', name=name)
  return tf.squeeze(pooled, axis=1)

def causal_linear(x, n_inputs, n_outputs, name, filter_length, rate, batch_size):
  # create queue
  q_1 = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, n_inputs))
  init_1 = q_1.enqueue_many(tf.zeros((rate, batch_size, n_inputs)))
  state_1 = q_1.dequeue()
  push_1 = q_1.enqueue(x)

  if filter_length == 3:
    q_2 = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, n_inputs))
    init_2 = q_2.enqueue_many(tf.zeros((rate, batch_size, n_inputs)))
    state_2 = q_2.dequeue()
    push_2 = q_2.enqueue(x)  

  w = tf.get_variable(
      name=name + "/W",
      shape=[1, filter_length, n_inputs, n_outputs],
      dtype=tf.float32)
  b = tf.get_variable(
      name=name + "/biases", shape=[n_outputs], dtype=tf.float32)

  if filter_length == 3:
    w_q_2 = w[0, 0, :, :]
    w_q_1 = w[0, 1, :, :]
    w_x = w[0, 2, :, :]
    y = tf.nn.bias_add(
      tf.matmul(state_2, w_q_2) + tf.matmul(state_1, w_q_1) + tf.matmul(x, w_x), b)
    return y, (init_1, init_2), (push_1, push_2)
  else:
    w_q_1 = w[0, 0, :, :]
    w_x = w[0, 1, :, :]   
    y = tf.nn.bias_add(
      tf.matmul(state_1, w_q_1) + tf.matmul(x, w_x), b)
    return y, (init_1, ), (push_1, )

def linear(x, n_inputs, n_outputs, name):
  n_inputs = x.get_shape().as_list()[-1]
  w = tf.get_variable(name=name + "/W", 
    shape=[1, 1, n_inputs, n_outputs], dtype=tf.float32)
  b = tf.get_variable(name=name + "/biases", 
    shape=[n_outputs], dtype=tf.float32)
  y = tf.nn.bias_add(tf.matmul(x, w[0, 0]), b)
  return y

