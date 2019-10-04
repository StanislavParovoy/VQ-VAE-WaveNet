import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np


class VQVAE(object):


    def __init__(self, args):
        self.x = args['x']
        self.h = args['speaker']
        self.encoder = args['encoder']
        self.decoder = args['decoder']
        self.latent_dim = args['latent_dim']
        self.k = args['k']
        self.beta = args['beta']
        self._print = lambda s, t: print(s, t.shape) if args['verbose'] else None

        self._print('input x:', self.x)
        if self.h is not None:
            self._print('input h:', self.h)


    def _build_prior(self):
        with tf.name_scope('prior'):
            return
            low, high = tf.zeros(self.latent_dim), tf.ones(self.latent_dim)
            self.prior = tfp.distributions.Uniform(low=low, high=high)


    def _build_encoder(self):
        self.z_e = self.encoder.build(self.x)
        self._print('z_e:', self.z_e)


    def _build_embedding_space(self):
        # self.embedding = tf.Variable(tf.random.uniform([self.k, self.latent_dim]))
        self.embedding = tf.get_variable(name='embedding', 
                                         shape=[self.k, self.latent_dim], 
                                         initializer=tf.uniform_unit_scaling_initializer())
        self._print('embedding:', self.embedding)


    def _discretise(self):
        # z_e: [..., d] -> [..., 1, d]
        # embedding:            [k, d]
        expanded_ze = tf.expand_dims(self.z_e, -2)
        distances = tf.reduce_sum((expanded_ze - self.embedding) ** 2, axis=-1)

        # q(z|x) refers to the 2d grid in the middle in figure 1
        self.q_z_x = tf.argmin(distances, axis=-1)
        self._print('q(z|x):', self.q_z_x)
        self.e_k = tf.gather(params=self.embedding, indices=self.q_z_x)

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
        input_t = tf.placeholder(tf.float32, [None, 1])
        channels = self.z_q.shape.as_list()[-1]
        local_condition_t = tf.placeholder(tf.float32, [None, channels])
        self.decoder.build_generator(input_t=input_t,
                                     local_condition_t=local_condition_t,
                                     global_condition_t=self.h)


    def _compute_loss(self):
        '''
        name   reconstruction          vq                commitment     
        loss = log(p(x|z_q) + l2_error(sg[z_e(x)] - e) + beta * l2_error(z_e(x) - sg[e])
        grad   encoder, decoder        embedding         encoder
        '''

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=self.labels,
                                    logits=self.x_z_q)
        self.reconstruction_loss = tf.reduce_mean(loss, axis=0)
        tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)

        self.vq_loss = tf.reduce_mean((tf.stop_gradient(self.z_e) - self.e_k) ** 2)
        tf.summary.scalar('vq_loss', self.vq_loss)

        self.commitment_loss = self.beta * \
            tf.reduce_mean((self.z_e - tf.stop_gradient(self.e_k)) ** 2)
        tf.summary.scalar('commitment_loss', self.commitment_loss)
        self.summary = tf.summary.merge_all()


    def _build_optimiser(self, learning_rate_schedule):
        assert None not in [self.reconstruction_loss, self.vq_loss, self.commitment_loss]

        # self.global_step = tf.train.get_or_create_global_step()
        self.global_step = tf.Variable(tf.constant(0, dtype=tf.int32), trainable=False)
        self.lr = tf.Variable(learning_rate_schedule[0])
        for key, value in learning_rate_schedule.items():
            self.lr = tf.cond(
                tf.less(self.global_step, key), lambda: self.lr, lambda: tf.constant(value))
        optimiser = tf.train.AdamOptimizer(self.lr)

        # optimise reconstruction loss on encoder and decoder
        encoder_variables = tf.trainable_variables(scope='encoder')
        # decoder_variables = tf.trainable_variables(scope='decoder')
        reconstruction_op = optimiser.minimize(self.reconstruction_loss)
        # reconstruction_gradients = optimiser.compute_gradients(
            # self.reconstruction_loss, decoder_variables + encoder_variables)

        # optimise commitment loss on encoder
        commitment_gradients = optimiser.compute_gradients(
            self.commitment_loss, encoder_variables)

        # optimise vq loss on embedding space
        embedding_gradients = optimiser.compute_gradients(self.vq_loss, [self.embedding])

        self.train_op = [optimiser.apply_gradients(
            commitment_gradients + embedding_gradients,
            global_step=self.global_step), reconstruction_op]


    def _build(self):
        # self._build_prior()
        with tf.variable_scope('encoder'):
            self._build_encoder()
        with tf.variable_scope('embedding'):
            self._build_embedding_space()
            self._discretise()

    def build(self, learning_rate_schedule):
        self._build()
        with tf.variable_scope('decoder'):
            self._build_decoder()
        self._compute_loss()
        self._build_optimiser(learning_rate_schedule)


    def build_generator(self):
        # self._build_prior()
        self._build()
        with tf.variable_scope('decoder'):
            self._build_decoder_generator()


    def encode(self, sess, x, h):
        result = sess.run(self.encoding, {self.x: x, self.h: h})
        self._print('encoding shape:', result)
        return result

