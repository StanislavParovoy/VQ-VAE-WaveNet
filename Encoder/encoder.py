import tensorflow as tf


class Encoder():
  def __init__(self, scope, latent_dim, trainable=True):
    super(Encoder, self).__init__()
    self.scope = scope
    self.latent_dim = latent_dim
    self.trainable = trainable

  def __call__(self, x):
    with tf.variable_scope(self.scope):
      return self.call(x)

  def call(self, x):
    tkl = tf.keras.layers
    filters = 368
    for i in range(6):
      x = tkl.Conv1D(filters=2*filters,
               kernel_size=4, 
               strides=2,
               padding='same')(x)
      x = tf.nn.tanh(x[:, :, : filters]) * tf.nn.sigmoid(x[:, :, filters: ])
    x = tkl.Conv1D(filters=self.latent_dim,
             kernel_size=1, 
             strides=1, 
             padding='valid')(x)
    x = tkl.BatchNormalization()(x)
    tf.summary.histogram('z_e', x)
    return x


class Encoder_():
  def __init__(self, scope, latent_dim, trainable=True):
    super(Encoder_, self).__init__()
    self.scope = scope
    self.latent_dim = latent_dim
    self.trainable = trainable
    self.layers = []
    encoder_layers = [(2, 4, 1),
              (2, 4, 1),
              (2, 4, 1),
              (1, 4, 1),
              (2, 4, 1),
              (1, 4, 1),
              (2, 4, 1),
              (1, 4, 1),
              (2, 4, 1),
              (1, 4, 1)]
    for strides, kernel_size, dilations in encoder_layers:
      conv_wide = tkl.Conv1D(filters=2*latent_dim,
                 kernel_size=kernel_size, 
                 strides=strides,
                 padding='same',
                 activation='relu')
      conv_1x1 = 0
      self.layers.append((conv_wide, conv_1x1))
    self.post1 = 0
    self.post2 = 0

  def __call__(self, x):
    for conv_wide, conv_1x1, skip, strides in self.layers:
      x = conv_wide(x, trainable=trainable)
      x = tf.nn.tanh(x[:, :, : self.latent_dim]) * tf.nn.sigmoid(x[:, :, self.latent_dim: ])
      res = conv_1x1(x, trainable=trainable)
      x = res + x
    x = self.post1(x, trainable=trainable)
    x = tf.nn.relu(x)
    x = self.post2(x, trainable=trainable)
    tf.summary.histogram('z_e', x)
    return x

