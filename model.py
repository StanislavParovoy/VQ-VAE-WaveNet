import tensorflow as tf


class VQVAE(object):


    def __init__(self, args):
        self.x = args['x']
        self.h = args['speaker']
        self.encoder = args['encoder']
        self.decoder = args['decoder']
        self.k = args['k']
        self.beta = args['beta']
        self.use_vq = args['use_vq']
        self.num_speakers = args['num_speakers']
        self._print = lambda s, t: print(s, t.shape) if args['verbose'] else None

        self._print('input x:', self.x)
        if self.h is not None:
            k = args['speaker_embedding']
            if k > 0:
                self.h = tf.argmax(self.h, axis=-1)
                self.speaker_embedding = tf.get_variable(
                                            name='speaker_embedding', 
                                            shape=[self.num_speakers, k],
                                            initializer=tf.uniform_unit_scaling_initializer(2.))
                self.h = tf.nn.embedding_lookup(self.speaker_embedding, self.h)
                tf.summary.histogram('speaker_embedding', self.speaker_embedding)
                u, v = tf.nn.moments(self.speaker_embedding, [-1])
                tf.summary.histogram('speaker_embedding_u', u)
                tf.summary.histogram('speaker_embedding_v', v)
            self._print('input h:', self.h)
        self.ema = tf.train.ExponentialMovingAverage(decay=0.999)


    def _build_encoder(self):
        self.z_e = self.encoder.build(self.x)
        self._print('z_e:', self.z_e)
        u, v = tf.nn.moments(self.z_e, [-1])
        tf.summary.histogram('z_e_u', u)
        tf.summary.histogram('z_e_v', v)
        tf.summary.histogram('z_e', self.z_e)


    def _build_embedding_space(self):
        latent_dim = self.z_e.get_shape().as_list()[-1]
        self.embedding = tf.get_variable(name='embedding', 
                                         shape=[self.k, latent_dim], 
                                         initializer=tf.uniform_unit_scaling_initializer(1.7))
        u, v = tf.nn.moments(self.embedding, [-1])
        tf.summary.histogram('embedding_u', u)
        tf.summary.histogram('embedding_v', v)
        tf.summary.histogram('embedding', self.embedding)
        self._print('embedding:', self.embedding)


    def _discretise(self):
        # z_e: [..., d] -> [..., 1, d]
        # embedding:            [k, d]
        expanded_ze = tf.expand_dims(self.z_e, -2)
        distances = tf.reduce_sum((expanded_ze - self.embedding) ** 2, axis=-1)
        tf.summary.histogram('distances', distances)

        # q(z|x) refers to the 2d grid in the middle in figure 1
        self.q_z_x = tf.argmin(distances, axis=-1)
        tf.summary.histogram('q(z|x)', self.q_z_x)
        self._print('q(z|x):', self.q_z_x)
        self.e_k = tf.nn.embedding_lookup(self.embedding, self.q_z_x)
        tf.summary.histogram('e_k', self.e_k)

        # this should pass gradient from z_q to z_e
        # evaluates to: self.z_q = tf.gather(params=self.embedding, indices=self.q_z_x)
        self.z_q = self.z_e + tf.stop_gradient(self.e_k - self.z_e)
        self._print('z_q:', self.z_q)


    def _build_decoder(self):
        self.x_z_q, self.labels = self.decoder.build(x=self.x,
                                     local_condition=self.z_q,
                                     global_condition=self.h)
        self._print('x|z_q:', self.x_z_q)
        self._print('labels:', self.labels)


    def _build_decoder_generator(self):
        self.encoding = self.decoder.build_generator(local_condition=self.z_q,
                                                     global_condition=self.h)


    def _compute_loss(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                     labels=self.labels,
                     logits=self.x_z_q)
        self.reconstruction_loss = tf.reduce_mean(loss, axis=0)
        tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)

        self.loss = self.reconstruction_loss

        if self.use_vq:
            self.vq_loss = tf.reduce_mean((tf.stop_gradient(self.z_e) - self.e_k) ** 2)
            tf.summary.scalar('vq_loss', self.vq_loss)

            self.commitment_loss = self.beta * tf.reduce_mean((self.z_e - tf.stop_gradient(self.e_k)) ** 2)
            tf.summary.scalar('commitment_loss', self.commitment_loss)

            self.loss += self.vq_loss + self.commitment_loss


    def _build_optimiser(self, learning_rate_schedule):
        self.global_step = tf.Variable(tf.constant(0, dtype=tf.int32), trainable=False)
        self.lr = tf.Variable(learning_rate_schedule[0])
        for key, value in learning_rate_schedule.items():
            self.lr = tf.cond(
                tf.less(self.global_step, key), lambda: self.lr, lambda: tf.constant(value))

        self.optimiser = tf.train.AdamOptimizer(self.lr)

        # self.train_op = optimiser.minimize(self.loss, global_step=self.global_step)
        self.train_op = tf.contrib.layers.optimize_loss(
                        loss=self.loss,
                        global_step=self.global_step,
                        learning_rate=self.lr, 
                        optimizer=self.optimiser,
                        summaries=['gradients'])
    
        trainable_variables = tf.trainable_variables()
        with tf.control_dependencies([self.train_op]):
            self.train_op = self.ema.apply(trainable_variables)
    
        self.summary = tf.summary.merge_all()


    def _build(self):
        with tf.variable_scope('encoder'):
            self._build_encoder()
        with tf.variable_scope('embedding'):
            if self.use_vq:
                self._build_embedding_space()
                self._discretise()
            else:
                self.e_k = self.z_e
                self.z_q = self.z_e


    def build(self, learning_rate_schedule):
        self._build()
        with tf.variable_scope('decoder'):
            self._build_decoder()
        with tf.variable_scope('optimiser'):
            self._compute_loss()
            self._build_optimiser(learning_rate_schedule)


    def build_generator(self):
        self._build()
        with tf.variable_scope('decoder'):
            self._build_decoder_generator()
        with tf.variable_scope('optimiser'):
            self.ema.apply(tf.trainable_variables())

