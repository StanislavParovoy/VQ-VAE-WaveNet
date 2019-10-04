import tensorflow as tf
import numpy as np

def mu_law_encode(x, quantization_channels=256, to_int=False):
    mu = tf.cast(quantization_channels - 1, tf.float32)
    x = tf.clip_by_value(x, -1., 1.)
    y = tf.sign(x) * tf.math.log1p(mu * tf.abs(x)) / tf.math.log1p(mu)
    if to_int:
        # [-1, 1](float) -> (0, mu)(int); + 0.5 since tf.cast does flooring
        return tf.cast((y + 1) / 2 * mu + 0.5, tf.int32)
    return y


def mu_law_decode(y, quantization_channels=256):
    mu = tf.cast(quantization_channels - 1, tf.float32)
    # (0, mu) -> (-1, 1)
    y = (2 * tf.cast(y, tf.float32) / mu) - 1
    x = tf.sign(y) * ((1 + mu) ** tf.abs(y) - 1) / mu
    return x


def mu_law_decode_np(y, quantization_channels=256):
    mu = np.asarray(quantization_channels - 1, dtype=np.float32)
    # (0, mu) -> (-1, 1)
    y = (2 * np.asarray(y, dtype=np.float32) / mu) - 1
    x = np.sign(y) * ((1 + mu) ** abs(y) - 1) / mu
    return x
