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
      old = x
      x = tkl.Conv1D(filters=2*filters,
               kernel_size=4, 
               strides=2,
               padding='same')(x)
      x = tf.nn.tanh(x[:, :, : filters]) * tf.nn.sigmoid(x[:, :, filters: ])
      x += tkl.Conv1D(filters=filters,
                      kernel_size=1,
                      strides=2,
                      padding='valid')(old)
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
    self.layers = []
    tkl = tf.keras.layers
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
                 padding='valid',
                 trainable=trainable)
      conv_1x1 = tkl.Conv1D(filters=latent_dim,
                 kernel_size=1, 
                 strides=1,
                 padding='same',
                 trainable=trainable)
      skip = (kernel_size - strides) * dilations
      self.layers.append((conv_wide, conv_1x1, skip, strides))
    self.post1 = tkl.Conv1D(filters=latent_dim,
                 kernel_size=1, 
                 strides=1,
                 padding='same',
                 trainable=trainable)
    self.post2 = tkl.Conv1D(filters=latent_dim,
                 kernel_size=1, 
                 strides=1,
                 padding='same',
                 trainable=trainable)

  def __call__(self, x):
    with tf.variable_scope(self.scope):
      for i, (conv_wide, conv_1x1, skip, strides) in enumerate(self.layers):
        old = x
        x = conv_wide(x)
        x = tf.nn.tanh(x[:, :, : self.latent_dim]) * tf.nn.sigmoid(x[:, :, self.latent_dim: ])
        res = conv_1x1(x)
        if i == 0:
          x = res
        else:
          old = old[:, skip: skip+tf.shape(res)[1]*strides, :]
          old = tf.reshape(old, [tf.shape(old)[0], tf.shape(res)[1], -1, tf.shape(res)[2]])
          old = old[:, :, -1, :]
          x = res + old
      x = self.post1(x)
      x = tf.nn.relu(x)
      x = self.post2(x)
    tf.summary.histogram('z_e', x)
    return x

