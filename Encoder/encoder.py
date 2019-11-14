from Encoder.encoder_ops import *
from Decoder.WaveNet.wavenet_ops import *
from mu_law_ops import *
import tensorflow as tf
tkl = tf.keras.layers


class Encoder_64():
    def __init__(self, latent_dim):
        super(Encoder_64, self).__init__()
        self.latent_dim = latent_dim

    def build(self, net):
        net = mu_law_encode(net)
        net = shift_right(net)
        for i in range(6):
            net = tkl.Conv1D(filters=768,
                            kernel_size=4, 
                            strides=2,
                            padding='same',
                            activation='relu')(net)
            net = tkl.BatchNormalization()(net)
        net = tkl.Conv1D(filters=self.latent_dim,
                         kernel_size=1, 
                         strides=1, 
                         padding='valid')(net)
        net = tkl.BatchNormalization()(net)
        return net


class Encoder_Magenta():
    def __init__(self, latent_dim):
        super(Encoder_Magenta, self).__init__()
        self.latent_dim = latent_dim
        self.args = {}
        self.args['dilation_rates'] = [1, 2, 4, 8, 16, 16]
        self.args['num_cycles'] = 1
        self.args['num_cycle_layers'] = 6

    def build(self, net):
        filters = 128
        kernel_size = 5

        net = shift_right(net)
        net = mu_law_encode(net)

        with tf.variable_scope('preprocess'):
            en = conv1d_v2(net, filters, kernel_size, 'VALID', dilations=1, log=True)

        for i, dilations in enumerate(self.args['dilation_rates']):
            cycle_id = 'cycle_%d' % (1 + i // self.args['num_cycle_layers'])
            layer_id = 'layer_%d' % (1 + i % self.args['num_cycle_layers'])
            with tf.variable_scope(cycle_id + '/' + layer_id):
                with tf.variable_scope('dilated'):
                    d = conv1d_v2(en, filters, 1, 'VALID', 1, log=True, stride=2)
                with tf.variable_scope('gate'):
                    g = conv1d_v2(d, filters, kernel_size, 'VALID', dilations, log=True)
                with tf.variable_scope('filter'):
                    f = conv1d_v2(d, filters, kernel_size, 'VALID', dilations, log=True)
                gated = tf.nn.tanh(g) * tf.nn.sigmoid(f)
                with tf.variable_scope('residual'):
                    en = d + conv1d_v2(gated, filters, 1, 'VALID', log=True)
        with tf.variable_scope('postprocess'):
            en = conv1d_v2(en, self.latent_dim, 1, 'VALID', log=True)
        return en


class Encoder_2019():
    def __init__(self, latent_dim):
        super(Encoder_2019, self).__init__()
        self.latent_dim = latent_dim

        
    def build(self, net):
        # 1 fixed feature extraction layer
        net = mfcc(tf.squeeze(net, axis=-1))

        # 2 preprocessing conv with residual connections
        net = conv_3_768(net)
        conv = conv_3_768(net)
        net = conv + net

        # one downsampling strided conv
        net = strided_conv_4_768(net)

        # 2 conv with residual connections
        for _ in range(2):
            conv = conv_3_768(net)
            net = conv + net

        # 4 relu with residual connections
        # tbh, no idea what a 'relu layer' is referring to
        for _ in range(4):
            relu = conv_3_768(net)
            net = relu + relu

        # downsample to 64D to match embedding dimension in VQ-VAE
        net = linear_64(net, filters=self.latent_dim)

        return net
    
