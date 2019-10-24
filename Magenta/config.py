from masked import *
import tensorflow as tf

decay=1e-8
num_stages = 10
num_layers = 50
filter_length = 2
width = 256
skip_width = 512
ae_num_stages = 10
ae_num_layers = 6
initial_filter = 1
ae_filter_length = 5
ae_width = 128
ae_bottleneck_width = 64
k = 512

class FastGenerationConfig(object):
  def __init__(self, batch_size=1):
    self.batch_size = batch_size

  def add_gc(self, net, gc, filters):
    in_channels = gc.shape.as_list()[-1]
    kernel = tf.get_variable(name='kernel', 
                             shape=[1, in_channels, filters], 
                             dtype=tf.float32,
                             initializer=tf.uniform_unit_scaling_initializer(1.0))
    bias = tf.get_variable(name='bias', 
                           shape=[filters], 
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(1.0))
    net += tf.matmul(gc, kernel[0]) + bias
    return net    

  def build(self, inputs, gc):

    gc = tf.argmax(gc, axis=-1)
    self.speaker_emb = tf.get_variable(name='speaker_emb', shape=[109, ae_bottleneck_width])
    gc = tf.nn.embedding_lookup(self.speaker_emb, gc)

    # Encode the source with 8-bit Mu-Law.
    x = inputs
    batch_size = self.batch_size
    x_quantized = mu_law(x)
    x_scaled = x_quantized

    print('x:', x.shape)

    encoding = tf.placeholder(
        name='encoding', shape=[batch_size, ae_bottleneck_width], dtype=tf.float32)
    en = encoding

    init_ops, push_ops = [], []

    ###
    # The WaveNet Decoder.
    ###
    l = x_scaled
    l, inits, pushs = causal_linear(
        x=l,
        n_inputs=1,
        n_outputs=width,
        name='startconv',
        rate=1,
        batch_size=batch_size,
        filter_length=filter_length)

    for init in inits:
      init_ops.append(init)
    for push in pushs:
      push_ops.append(push)

    # Set up skip connections.
    s = linear(l, width, skip_width, name='skip_start')

    # Residual blocks with skip connections.
    for i in range(num_layers):
      dilation = 2**(i % num_stages)

      # dilated masked cnn
      d, inits, pushs = causal_linear(
          x=l,
          n_inputs=width,
          n_outputs=width * 2,
          name='dilatedconv_%d' % (i + 1),
          rate=dilation,
          batch_size=batch_size,
          filter_length=filter_length)

      for init in inits:
        init_ops.append(init)
      for push in pushs:
        push_ops.append(push)

      # local conditioning
      d += linear(en, ae_bottleneck_width, width * 2, name='cond_map_%d' % (i + 1))

      # global conditioning
      with tf.variable_scope('gc_%d'%(i+1)):
        d = self.add_gc(d, gc, width * 2)

      # gated cnn
      assert d.get_shape().as_list()[-1] % 2 == 0
      m = d.get_shape().as_list()[-1] // 2
      d = tf.sigmoid(d[:, :m]) * tf.tanh(d[:, m:])

      # residuals
      l += linear(d, width, width, name='res_%d' % (i + 1))

      # skips
      s += linear(d, width, skip_width, name='skip_%d' % (i + 1))

    s = tf.nn.relu(s)
    s = (linear(s, skip_width, skip_width, name='out1') + linear(
        en, ae_bottleneck_width, skip_width, name='cond_map_out1'))
    with tf.variable_scope("gc_final"):
        s = self.add_gc(s, gc, skip_width)
    s = tf.nn.relu(s)

    ###
    # Compute the logits and get the loss.
    ###
    logits = linear(s, skip_width, 256, name='logits')
    logits = tf.reshape(logits, [-1, 256])
    probs = tf.nn.softmax(logits, name='softmax')

#     ema = tf.train.ExponentialMovingAverage(decay=0.9996)
#     trainable_variables = tf.trainable_variables()
#     self.shadow = {ema.average_name(v): v for v in trainable_variables}

    return {
        'init_ops': init_ops,
        'push_ops': push_ops,
        'predictions': probs,
        'encoding': encoding,
        'quantized_input': x_quantized
    }

    
class Config(object):
  """Configuration object that helps manage the graph."""

  def __init__(self):
    self.learning_rate_schedule = {
        0: 0.0001,
        60000: 0.00008,
        80000: 0.00006,
        100000: 0.00004,
        120000: 0.00002,
        140000: 0.00001
    }
    self.ae_hop_length = 64

  @staticmethod
  def _condition(x, encoding):
    """Condition the input on the encoding.

    Args:
      x: The [mb, length, channels] float tensor input.
      encoding: The [mb, encoding_length, channels] float tensor encoding.

    Returns:
      The output after broadcasting the encoding to x's shape and adding them.
    """
    shape = tf.shape(x)
    mb = shape[0]
    length = shape[1]
    channels = x.shape.as_list()[-1]
    enc_length = tf.shape(encoding)[1]

    encoding = tf.expand_dims(encoding, 2)
    x = tf.reshape(x, [mb, enc_length, -1, channels])
    x += encoding
    x = tf.reshape(x, [mb, length, channels])
    return x

  def add_gc(self, net, gc, filters):
    in_channels = gc.shape.as_list()[-1]
    kernel = tf.get_variable(name='kernel', 
                             shape=[1, in_channels, filters], 
                             dtype=tf.float32,
                             initializer=tf.uniform_unit_scaling_initializer(1.0))
    bias = tf.get_variable(name='bias', 
                           shape=[filters], 
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(1.0))
    net += tf.nn.conv1d(gc, kernel, stride=1, padding='VALID') + bias
    return net

  def build(self, inputs, gc, is_training, rescale_inputs=True):

    # Encode the source with 8-bit Mu-Law.
    x = inputs
    x_quantized = mu_law(x)
    x_scaled = x_quantized # [-1, 1]
    print('x:', x.shape)

    ###
    # The Non-Causal Temporal Encoder.
    ###
    en = conv1d(
         x_scaled if rescale_inputs else x,
         causal=False,
         num_filters=ae_width,
         filter_length=initial_filter,
         name='ae_startconv',
         regularizer=tf.keras.regularizers.l2(decay),
         is_training=True)
    
    for num_layer in range(ae_num_layers):
        dilation = 2**(num_layer % ae_num_stages)
        dilation = 1
        conv = conv1d(
               en,
               causal=False,
               num_filters=ae_width,
               filter_length=ae_filter_length,
               dilation=dilation,
               name='ae_dilatedconv_%d' % (num_layer + 1),
               regularizer=tf.keras.regularizers.l2(decay),
               is_training=True)
        gate = conv1d(
               en,
               causal=False,
               num_filters=ae_width,
               filter_length=ae_filter_length,
               dilation=dilation,
               name='ae_dilatedgate_%d' % (num_layer + 1),
               regularizer=tf.keras.regularizers.l2(decay),
               is_training=True)
        d = tf.nn.tanh(conv) * tf.nn.sigmoid(gate)
        en += conv1d(
              d,
              num_filters=ae_width,
              filter_length=1,
              name='ae_res_%d' % (num_layer + 1),
              regularizer=tf.keras.regularizers.l2(decay),
              is_training=True)
        en = pool1d(en, 2, name='ae_pool_%d' % (num_layer + 1), mode='avg')
    en = conv1d(
         en,
         num_filters=ae_bottleneck_width,
         filter_length=1,
         name='ae_bottleneck',
         regularizer=tf.keras.regularizers.l2(decay),
         is_training=True)
#     en = pool1d(en, self.ae_hop_length, name='ae_pool', mode='avg')
    self.encoding = en
    print('en:', en.shape)
    
#     '''
    self.k = k
    self.latent_dim = ae_bottleneck_width
    z_e = en
    self.embedding = tf.get_variable(name='embedding', 
                                 # initializer = tf.eye(self.latent_dim))
                                 shape=[self.k, self.latent_dim], 
                                 # initializer=tf.initializers.orthogonal(gain=1.0),
                                 initializer=tf.uniform_unit_scaling_initializer(),
                                 regularizer=tf.keras.regularizers.l2(decay))
    expanded_ze = tf.expand_dims(z_e, -2)
    distances = tf.reduce_sum((expanded_ze - self.embedding) ** 2, axis=-1)
    q_z_x = tf.argmin(distances, axis=-1)
    e_k = tf.nn.embedding_lookup(self.embedding, q_z_x)
    z_q = z_e + tf.stop_gradient(e_k - z_e)
    en = z_q
    self.encoding = e_k

    self.vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - e_k) ** 2)
    tf.summary.scalar('vq_loss', self.vq_loss)
#     self.vq_loss = 0

    self.commitment_loss = 0.25 * tf.reduce_mean((z_e - tf.stop_gradient(e_k)) ** 2)
    tf.summary.scalar('commitment_loss', self.commitment_loss)
#     '''
#     self.commitment_loss, self.vq_loss = 0, 0

    if not is_training:
      return e_k

    gc = tf.argmax(gc, axis=-1)
    self.speaker_emb = tf.get_variable(name='speaker_emb', 
                             shape=[109, ae_bottleneck_width], 
                             dtype=tf.float32,
                             initializer=tf.uniform_unit_scaling_initializer(1.0))
    gc = tf.nn.embedding_lookup(self.speaker_emb, gc)
    print('gc:', gc.shape) # [b, 1]

    ###
    # The WaveNet Decoder.
    ###

    l = shift_right(x_scaled if rescale_inputs else x)
    l = conv1d(
        l,
        num_filters=width,
        filter_length=filter_length,
        name='startconv',
        regularizer=tf.keras.regularizers.l2(decay),
        is_training=is_training)

    # Set up skip connections.
    s = conv1d(
        l,
        num_filters=skip_width,
        filter_length=1,
        name='skip_start',
        regularizer=tf.keras.regularizers.l2(decay),
        is_training=is_training)

    # Residual blocks with skip connections.
    for i in range(num_layers):
      dilation = 2**(i % num_stages)
      d = conv1d(
          l,
          num_filters=2 * width,
          filter_length=filter_length,
          dilation=dilation,
          name='dilatedconv_%d' % (i + 1),
          regularizer=tf.keras.regularizers.l2(decay),
          is_training=is_training)
      if en is not None:
        d = self._condition(d,
                            conv1d(
                                en,
                                num_filters=2 * width,
                                filter_length=1,
                                name='cond_map_%d' % (i + 1),
                                is_training=is_training))
      if gc is not None:
        with tf.variable_scope('gc_%d'%(i+1)):
          d = self.add_gc(d, gc, filters=2*width)

      m = d.get_shape().as_list()[2] // 2
      d_sigmoid = tf.sigmoid(d[:, :, :m])
      d_tanh = tf.tanh(d[:, :, m:])
      d = d_sigmoid * d_tanh

      l += conv1d(
           d,
           num_filters=width,
           filter_length=1,
           name='res_%d' % (i + 1),
           regularizer=tf.keras.regularizers.l2(decay),
           is_training=is_training)
      s += conv1d(
           d,
           num_filters=skip_width,
           filter_length=1,
           name='skip_%d' % (i + 1),
           regularizer=tf.keras.regularizers.l2(decay),
           is_training=is_training)

    s = tf.nn.relu(s)
    s = conv1d(
        s,
        num_filters=skip_width,
        filter_length=1,
        name='out1',
        regularizer=tf.keras.regularizers.l2(decay),
        is_training=is_training)

    if en is not None:
      s = self._condition(
                s,
                conv1d(
                  en,
                  num_filters=skip_width,
                  filter_length=1,
                  name='cond_map_out1',
                  is_training=is_training))
    if gc is not None:
      with tf.variable_scope('gc_final'):
        s = self.add_gc(s, gc, filters=skip_width)
    s = tf.nn.relu(s)

    ###
    # Compute the logits and get the loss.
    ###
    logits = conv1d(
             s,
             num_filters=256,
             filter_length=1,
             name='logits',
             regularizer=tf.keras.regularizers.l2(decay),
             is_training=is_training)
    logits = tf.reshape(logits, [-1, 256])
    x_indices = tf.reshape(tf.cast((x_quantized + 1) / 2 * 255 + 0.5, tf.int32), [-1])
    self.reconstruction_loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=x_indices, name='nll'),
      0,
      name='loss')
    tf.summary.scalar('reconstruction loss', self.reconstruction_loss)

    reg_loss = tf.reduce_sum(tf.get_collection(
      tf.GraphKeys.REGULARIZATION_LOSSES)
    )
    tf.summary.scalar('regularization loss', reg_loss)
    self.loss = self.reconstruction_loss + reg_loss + self.vq_loss + self.commitment_loss
    self.global_step = tf.Variable(tf.constant(0, dtype=tf.int32), trainable=False)
    self.lr = tf.constant(self.learning_rate_schedule[0])
    for key, value in self.learning_rate_schedule.items():
      self.lr = tf.cond(
        tf.less(self.global_step, key), lambda: self.lr, lambda: tf.constant(value))
    self.opt = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(
      self.loss, global_step=self.global_step)

#     ema = tf.train.ExponentialMovingAverage(decay=0.995)
#     trainable_variables = tf.trainable_variables()
#     self.variables = {ema.average_name(v): v for v in trainable_variables}
#     self.variables[self.global_step.name] = self.global_step
#     with tf.control_dependencies([opt]):
#       self.opt = ema.apply(trainable_variables)

    self.summary = tf.summary.merge_all()

