import tensorflow as tf
import tensorflow_probability as tfp
import parameters
from Decoder.WaveNet.wavenet_ops import *

class WaveNet():
  def __init__(self, args, scope, trainable=True):
    self.receptive_field = 1 + sum(args['dilations']) * (args['kernel_size'] - 1)
    print('wavenet %s receptive_field: %d / %d' % (scope, self.receptive_field, parameters.sr))
    self.scope = scope
    self.trainable = trainable
    self.args = args

  def __call__(self, x, condition, h=None):
    ''' the wavenet decoder
    args:
      x: original raw audio, shape [b, t, 1]
      condition: spectrogram concatenated with speaker embedding
    returns:
      wavenet output, same shape as x
    '''  
    if h is not None:
      gc_dim = h.get_shape().as_list()[-1]
      h = tf.argmax(h, axis=-1)
      self.gc_embedding = tf.get_variable(
          name=self.scope + '/gc_embedding', 
          shape=[gc_dim, parameters.global_embedding_dim],
          initializer=tf.uniform_unit_scaling_initializer(2.))
      h = tf.nn.embedding_lookup(self.gc_embedding, h)
      h = tf.tile(h, [1, condition.get_shape().as_list()[1], 1])
      condition = tf.concat([condition, h], axis=-1)

    with tf.variable_scope(self.scope):
      return self.call(x, condition)

  def call(self, x, condition): 
    x = shift_right(x)

    # preprocess layer
    with tf.variable_scope('preprocess'):
      x = conv1d(x, filters=self.args['residual_filters'], kernel_size=1, trainable=self.trainable)

    skip_sum = []

    # residual stacks
    kernel_size = self.args['kernel_size']
    dilation_filters = self.args['dilation_filters']
    skip_filters = self.args['skip_filters']
    residual_filters = self.args['residual_filters']

    for i, dilations in enumerate(self.args['dilations']):
      with tf.variable_scope('layer_%d' % (i + 1)):
        skip, res = residual_stack(x, dilation_filters, kernel_size, dilations, \
          skip_filters, residual_filters, condition, trainable=self.trainable)
        skip_sum.append(skip)
        x += res
    x = sum(skip_sum)

    # postprocess layer 1 with condition
    with tf.variable_scope('postprocess1'):
      x = tf.nn.relu(x)
      x = conv1d(x, filters=self.args['skip_filters'], kernel_size=1, trainable=self.trainable)

    # postprocess layer 2, outputs logits (mean, log_std)
    filters = 2
    with tf.variable_scope('postprocess2'):
      x = tf.nn.relu(x)
      x = conv1d(x, filters=filters, kernel_size=1, trainable=self.trainable)
    
    print('logits:', x.shape)
    return x

  def compute_loss(self, logits, labels):
    logits = tf.reshape(logits, [-1, tf.shape(logits)[-1]])
    labels = tf.reshape(labels, [-1])
    u = logits[:, 0]
    log_s = tf.maximum(logits[:, 1], self.args['min_log_scale'])
    s = tf.exp(log_s)
    D = tfp.distributions.Normal(loc=u, scale=s)
    log_prob = D.log_prob(labels)
    nnl = -tf.reduce_mean(log_prob)
    tf.summary.scalar('nnl', nnl)
    tf.summary.histogram('log_s', log_s)
    print('labels:', labels.shape)
    print('log_prob:', log_prob.shape)
    print('loss:', nnl.shape)
    return nnl
    '''
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
               labels=labels,
               logits=logits)
    loss = tf.reduce_mean(loss, axis=0)
    '''

  def generate(self, input_t, condition_t, batch_size):
    ''' performs the fast wavenet generation for one step
    args:
        input_t: initial value (at first time stamp)
        condition_t: initial value of condition_t (at first time stamp)
    returns:
        value at final layer of wavenet (at second time stamp)
    '''
    with tf.variable_scope(self.scope):
      return self._generate(input_t, condition_t, batch_size)

  def _generate(self, input_t, condition_t, batch_size):
    init_ops = []
    push_ops = []

    # preprocess layer
    with tf.variable_scope('preprocess'):
      x, inits, pushes = fast_conv1d(input_t, self.args['residual_filters'], 1, 1, batch_size)
      init_ops.extend(inits)
      push_ops.extend(pushes)

    skip_sum = []

    # residual stacks
    kernel_size = self.args['kernel_size']
    state_size = self.args['dilation_filters']
    skip_filters = self.args['skip_filters']
    residual_filters = self.args['residual_filters']

    for i, dilations in enumerate(self.args['dilations']):
      with tf.variable_scope('layer_%d' % (i + 1)):
        skip, res, inits, pushes = fast_residual_stack(
            x, state_size, kernel_size, dilations, batch_size, \
            condition_t, skip_filters, residual_filters)
        skip_sum.append(skip)
        x += res
        init_ops.extend(inits)
        push_ops.extend(pushes)

    x = sum(skip_sum)

    # postprocess layer 1 with condition
    with tf.variable_scope('postprocess1'):
      x = tf.nn.relu(x)
      x = linear(x, self.args['skip_filters'])

    # postprocess layer 2, outputs logits
    with tf.variable_scope('postprocess2'):
      x = tf.nn.relu(x)
      # x = linear(x, 3 * self.args['num_logistics'])
      x = linear(x, 2)

    return init_ops, push_ops, x

