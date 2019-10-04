from Encoder.encoder_ops import *
from Decoder.WaveNet.wavenet_ops import *
from mu_law_ops import *
import tensorflow as tf
tkl = tf.keras.layers


class Encoder_64():
    def __init(self):
        super(Encoder_64, self).__init__()

    def build(self, net):
        latent_dim = 512
        net = mu_law_encode(net)
        for i in range(6):
            net = tkl.Conv1D(filters=768,
                            kernel_size=4, 
                            strides=2,
                            padding='same',
                            activation='relu')(net)
        net = tkl.Conv1D(filters=latent_dim, kernel_size=1, strides=1, padding='valid')(net)
        print('en:', net.shape)
        return net


class Encoder_Magenta():
    def __init__(self):
        super(Encoder_64, self).__init__()
        self.args = {}
        self.args['dilation_rates'] = [1,2,4,8,16,32,64,128,256,512,
                                       1,2,4,8,16,32,64,128,256,512,
                                       1,2,4,8,16,32,64,128,256,512]
        self.args['num_cycles'] = 3
        self.args['num_cycle_layers'] = 10

    def build(self, net):
        filters = 128
        kernel_size = 3
        latent_dim = 64
        hop_length = 256

        net = mu_law_encode(net)

        with tf.variable_scope('preprocess'):
            en = conv1d_v2(net, filters, kernel_size, 'VALID', dilations=1)

        for i, dilations in enumerate(self.args['dilation_rates']):
            cycle_id = 'cycle_%d' % (1 + i // self.args['num_cycle_layers'])
            layer_id = 'layer_%d' % (1 + i % self.args['num_cycle_layers'])
            with tf.variable_scope(cycle_id + '/' + layer_id):
                print('dr_%d: d: ' % dilations, end='')
                d = tf.nn.relu(en)
                with tf.variable_scope('dilated'):
                    d = conv1d_v2(d, filters, kernel_size, 'VALID', dilations)
                d = tf.nn.relu(d)
                print(d.shape, '  en:', en.shape)
                with tf.variable_scope('residual'):
                    en = en + conv1d_v2(d, filters, 1, 'VALID')
        with tf.variable_scope('postprocess'):
            en = conv1d_v2(en, latent_dim, 1, 'VALID')
            en = pool1d(en, hop_length)
        print('en:', en.shape)
        return en


class Encoder2019():
    def __init__(self):
        super(Encoder2019, self).__init__()

        
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
        net = linear_64(net)

        return net
    
