import tensorflow as tf
import numpy as np

class Upsample:
  def __init__(self, scope='upsample', strides=[15, 20], mode='convT', trainable=True):
    super(Upsample, self).__init__()
    self.scope = scope
    self.strides = strides
    self.mode = mode
    self.trainable = trainable

  def __call__(self, y, length=None):
    with tf.variable_scope(self.scope):
      return self.call(y, length)

  def call(self, y, length=None):
    y = tf.expand_dims(y, 1)
    if self.mode == 'convT':
      filters = y.get_shape().as_list()[-1]
      for i, s in enumerate(self.strides):
        y = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(3, s * 2),
            strides=(1, s),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.4),
            use_bias=True,
            kernel_initializer='glorot_normal',
            trainable=self.trainable)(y)
    else:
      y = tf.image.resize(y, [1, length], method='nearest')
    y = tf.squeeze(y, 1)
    return y

def shift_right(x, constant_values=0):
  shape = x.get_shape().as_list()
  x_padded = tf.pad(x, [[0, 0], [1, 0], [0, 0]], constant_values=constant_values)
  x_sliced = x_padded[:, :-1, :]
  x_sliced.set_shape(shape)
  return x_sliced

def conv1d(x, 
           filters,
           kernel_size=1,
           padding='valid',
           dilation_rate=1,
           use_bias=True,
           dropout=0.05,
           trainable=True): 
  conv = tf.keras.layers.Conv1D(
    filters,
    kernel_size,
    strides=1,
    padding=padding,
    dilation_rate=dilation_rate,
    activation=None,
    use_bias=use_bias,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    trainable=trainable)
  x = tf.pad(x, [[0, 0], [dilation_rate * (kernel_size - 1), 0], [0, 0]])
  x = conv(x)
  if trainable:
    x = tf.nn.dropout(x, rate=dropout)
  return x

def conv1d_v2(x, 
           filters,
           kernel_size=1,
           padding='VALID',
           dilation_rate=1,
           use_bias=True,
           dropout=0.05,
           trainable=True,
           log=True): 
  in_channels = x.shape.as_list()[-1]
  shape = [kernel_size, in_channels, filters]
  kernel = tf.get_variable(name='kernel', 
                           shape=shape, 
                           dtype=tf.float32,
                           initializer=tf.keras.initializers.orthogonal,
                           trainable=trainable)
  if use_bias:
    bias = tf.get_variable(name='bias', 
                           shape=[filters], 
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(0.0),
                           trainable=trainable)
  
  x = tf.pad(x, [[0, 0], [dilation_rate * (kernel_size - 1), 0], [0, 0]])
  kernel = tf.expand_dims(kernel, axis=0) # [fw, in, out] -> [fh=1, fw, in, out]
  stride = [1, 1, 1, 1]
  dilation_rate = [1, 1, dilation_rate, 1]
  x = tf.expand_dims(x, axis=1) # [b, t, c] -> [b, h=1, t, c]
  x = tf.nn.conv2d(x, kernel, strides=stride, padding=padding, dilations=dilation_rate)
  x = tf.squeeze(x, axis=1)
  if use_bias:
    x += bias
  if trainable:
    x = tf.nn.dropout(x, rate=dropout)
  return x

def add_condition(x, condition, trainable=True):
  if condition is not None:
    _, x_len, out_channels = x.shape.as_list()
    T = condition.shape.as_list()[1]
    encoding = conv1d(condition, 
                      filters=out_channels, 
                      kernel_size=1, 
                      use_bias=False,
                      trainable=trainable)
    # assert T == x_len, "condition length mismatch"
    # return x + encoding
    x = tf.reshape(x, [-1, T, x_len // T, out_channels])
    x += tf.expand_dims(encoding, 2)
    x = tf.reshape(x, [-1, x_len, out_channels])
  return x

def residual_stack(x, dilation_filters, kernel_size, dilation_rate, 
  skip_filters, residual_filters, condition=None, trainable=True):
  ''' Performs one layer of residual stack, with skip and residual connections
  Args:
    x: tensor of shape [b, t, c]
    dilation_rate: dilation rate
    *filters: out channels for that conv op
    local_condition: upsampled output from VQ-VAE, same resolution as x
  Returns:
    skip, residual, both results of 1x1 conv
  '''
  with tf.variable_scope('gated'):
    x = conv1d(x, filters=2*dilation_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, trainable=trainable)
    with tf.variable_scope('condition'):
      x = add_condition(x, condition, trainable=trainable)
    x_filter, x_gate = x[:, :, : dilation_filters], x[:, :, dilation_filters: ]
    x = tf.nn.tanh(x_filter) * tf.nn.sigmoid(x_gate)

  with tf.variable_scope('skip'):
    skip = conv1d(x, filters=skip_filters, kernel_size=1, trainable=trainable)

  with tf.variable_scope('residual'):
    residual = conv1d(x, filters=residual_filters, kernel_size=1, trainable=trainable)

  return skip, residual

'''
below is the fast wavenet autoregressive generation, based on:
https://arxiv.org/abs/1611.09482
https://github.com/ibab/tensorflow-wavenet
'''

def linear(x, filters, use_bias=True):
  ''' Performs one stride of a 1x1 convolution
  Args: 
    x: the current state
    filters: number of filters of result
  Returns:
    tensor of shape [b, filters]
  '''
  in_channels = x.shape.as_list()[-1]
  # kernel = tf.get_variable(name='kernel', shape=[1, in_channels, filters])

  dum = tf.ones([1,5120,in_channels])
  l = tf.keras.layers.Conv1D(
    filters,
    1,
    strides=1,
    padding='valid',
    data_format='channels_last',
    dilation_rate=1,
    use_bias=use_bias
  )
  dum = l(dum)
  kernel = l.weights[0]

  if use_bias:
    # bias = tf.get_variable(name='bias', shape=[filters])
    bias = l.weights[1]
    return tf.matmul(x, kernel[0]) + bias
  return tf.matmul(x, kernel[0])

def fast_conv1d(current, filters, kernel_size, dilation_rate, batch_size):
  ''' performs one stride of convolution on a layer
  Args:
    see fst_gated_cnn
  Returns:
    a new state at t+1
    init_ops from fast_conv1d
    push_ops from fast_conv1d
  '''
  in_channels = current.shape.as_list()[-1]
  kernel_shape = [kernel_size, in_channels, filters]
  # kernel = tf.get_variable(name='kernel', shape=kernel_shape)
  # bias = tf.get_variable(name='bias', shape=[filters])

  dum = tf.ones([1,5120,in_channels])
  l = tf.keras.layers.Conv1D(
    filters,
    kernel_size,
    strides=1,
    padding='valid',
    data_format='channels_last',
    dilation_rate=dilation_rate,
    activation=None,
    use_bias=True
  )
  dum = l(dum)
  kernel, bias = l.weights

  init_ops, push_ops = [], []
  new_state = tf.matmul(current, kernel[kernel_size - 1]) + bias
  state_size = in_channels
  for i in range(1, kernel_size):
    q = tf.FIFOQueue(dilation_rate,
             dtypes=tf.float32,
             shapes=(batch_size, state_size))
    init = q.enqueue_many(tf.zeros((dilation_rate, batch_size, state_size)))

    # dequeue past, enqueue state dequeued from last queue
    past = q.dequeue()
    push = q.enqueue([current])
    current = past
    init_ops.append(init)
    push_ops.append(push)

    new_state += tf.matmul(past, kernel[kernel_size - i - 1])

  return new_state, init_ops, push_ops

def fast_condition(x, condition_t):
  ''' Adds condition to x. 
  Args:
    x: the current state at time t
    condition_t: local / global condition at time t
  Returns:
    x added with 1x1 condition_t
  '''
  if condition_t is not None:
    out_channels = x.shape.as_list()[-1]
    x += linear(condition_t, out_channels, use_bias=False)
  return x

def fast_gated_cnn(current, dilation_filters, kernel_size, 
  dilation_rate, batch_size, conditino_t):
  ''' Performs one stride of the gated convolution. 
  Args:
    current: the current state at time t
    dilation_filters: number of filters for dilated_causal_conv
    kernel_size: filter width of dilated_causal_conv
    dilation_rate: 
    batch_size: will determine size of queue
    conditino_t: local condition at time t, placeholder
  Returns:
    gated conv on x
    init_ops from fast_conv1d
    push_ops from fast_conv1d
  '''
  x, init_ops, push_ops = \
    fast_conv1d(current, 2 * dilation_filters, kernel_size, dilation_rate, batch_size)
  with tf.variable_scope('condition'):
    x = fast_condition(x, conditino_t)
  # half of gated to tanh, half of gated to sigmoid
  net_filter, net_gate = x[:, : dilation_filters], x[:, dilation_filters: ]
  x = tf.nn.tanh(net_filter) * tf.nn.sigmoid(net_gate)
  return x, init_ops, push_ops

def fast_residual_stack(current, dilation_filters, kernel_size, dilation_rate, batch_size,
  conditino_t, skip_filters, residual_filters):
  ''' Performs one stride of one layer of residual stack
  Args:
    current: current state at time t
    dilation_filters:
    kernel_size:
    dilation_rate:
    batch_size: for fast_conv1d
    skip_filters: number of filters for skip
    residual_filters: number of filters for residual
  Returns:
    skip connection of shape [b, skip_filters] 
    residual of shape [b, residual_filters]
    init_ops from fast_conv1d
    push_ops from fast_conv1d
  '''
  with tf.variable_scope('gated'):
    gated, init_ops, push_ops = fast_gated_cnn(current, dilation_filters, \
      kernel_size, dilation_rate, batch_size, conditino_t)

  with tf.variable_scope('skip'):
    skip = linear(gated, skip_filters)

  with tf.variable_scope('residual'):
    residual = linear(gated, residual_filters)

  return skip, residual, init_ops, push_ops

