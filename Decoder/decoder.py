import tensorflow as tf
from Decoder.decoder_ops import *
from Decoder.WaveNet.wavenet import Wavenet

        
class WavenetDecoder():
    def __init__(self, args_file):
        super(WavenetDecoder, self).__init__()
        self.args_file = args_file


    def build(self, x, local_condition, global_condition, is_training=True):
        ''' decodes autoregressively using wavenet
        args:
            net: original audio
            local_condition: VQ-VAE output
            global_condition: [B, 1, NUM_SPEAKERS]
        return:
            x|z_q
        '''

        '''
        # ops as in the 2019 paper
        if is_training:
            local_condition = time_jitter(local_condition)
        local_condition = conv_3_128(local_condition)
        local_condition = upsample(local_condition, size=256)
        '''

        if global_condition is not None:
            local_condition = concat(local_condition, global_condition)

        self.wavenet = Wavenet(self.args_file)
        logits, labels = self.wavenet.build(inputs=x, 
                                            local_condition=local_condition, 
                                            global_condition=None)
        return logits, labels


    def build_generator(self, local_condition, global_condition):
        ''' decodes autoregressively using wavenet
        args:
            local_condition: VQ-VAE output
            global_condition: [B, 1, NUM_SPEAKERS]
        return:
            x|z_q
        '''

        if global_condition is not None:
            local_condition = concat(local_condition, global_condition)

        batch_size, _, channels = local_condition.get_shape().as_list()
        input_t = tf.placeholder(tf.int32, [batch_size])
        local_condition_t = tf.placeholder(tf.float32, [batch_size, channels])

        self.wavenet = Wavenet(self.args_file)
        self.wavenet.build_generator(input_t=input_t, 
                                     local_condition_t=local_condition_t, 
                                     global_condition_t=None, 
                                     batch_size=batch_size)

        return local_condition

