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
        if is_training:
            local_condition = time_jitter(local_condition)

        local_condition = conv_3_128(local_condition)
        print('local_condition:', local_condition.shape)
        local_condition = upsample(local_condition, size=256)
        print('upsample:', local_condition.shape)
        '''
        
        # if global_condition is not None:
        #     print('global_condition:', global_condition.shape)
        #     local_condition = concat(local_condition, global_condition)
        # print('condition::', local_condition.shape)

        self.wavenet = Wavenet(self.args_file)
        logits, labels = self.wavenet.build(inputs=x, 
                                            local_condition=local_condition, 
                                            global_condition=global_condition,
                                            batch_size=x.get_shape().as_list()[0])
        return logits, labels


    def build_generator(self, input_t, local_condition_t, global_condition_t):
        ''' decodes autoregressively using wavenet
        args:
            net: original audio
            local_condition: VQ-VAE output
            global_condition: [B, 1, NUM_SPEAKERS]
        return:
            x|z_q
        '''

        self.wavenet = Wavenet(self.args_file)
        self.wavenet.build_generator(input_t, local_condition_t, global_condition_t, 1)

