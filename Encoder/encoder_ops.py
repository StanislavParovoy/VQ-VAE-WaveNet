import tensorflow as tf
tkl = tf.keras.layers


def pool1d(net, hop_length):
    net = tf.expand_dims(net, 1)
    ksize = [1, 1, hop_length, 1]
    strides = [1, 1, hop_length, 1]
    net = tf.nn.avg_pool(net, ksize, strides, 'SAME')
    net = tf.squeeze(net, 1)
    return net


def mfcc(batch_wav):
    sr = 16000
    frame_length = int(25 / 1000 * sr) # 25 ms window
    frame_step = int(10 / 1000 * sr)   # every 10 ms
    num_mel_bins = 80                  # 80 features
    lower_edge_hertz = 20.0
    upper_edge_hertz = 8000.0

    stft = tf.abs(tf.contrib.signal.stft(signals=batch_wav, 
                                frame_length=frame_length,
                                frame_step=frame_step,
                                fft_length=frame_length,
                                window_fn=tf.contrib.signal.hann_window,
                                pad_end=True))

    num_spectrogram_bins = stft.shape[-1]
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
                                num_mel_bins=num_mel_bins,
                                num_spectrogram_bins=num_spectrogram_bins,
                                sample_rate=sr,
                                lower_edge_hertz=lower_edge_hertz,
                                upper_edge_hertz=upper_edge_hertz,
                                dtype=tf.float32)
    feature = tf.tensordot(stft, linear_to_mel_weight_matrix, 1)
    feature.set_shape(stft.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    print('feature:', feature.shape)

    feature = tf.math.log(feature + 1e-6)
    feature = tf.contrib.signal.mfccs_from_log_mel_spectrograms(feature)[..., :13]
    return feature


def conv_3_768(net, relu='relu'):
    net = tkl.Conv1D(filters=768,
                    kernel_size=3, 
                    strides=1,
                    padding='same',
                    activation=relu)(net)
    return net


def strided_conv_4_768(net, relu='relu'):
    net = tkl.Conv1D(filters=768,
                    kernel_size=4, 
                    strides=2,
                    padding='same',
                    activation=relu)(net)
    return net    


def linear_64(net, filters=64):
    # using a 1x1 conv to downsample
    net = tkl.Conv1D(filters=filters,
                    kernel_size=1, 
                    strides=1,
                    padding='same')(net)
    return net

