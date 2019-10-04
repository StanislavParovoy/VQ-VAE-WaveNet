import tensorflow as tf
tkl = tf.keras.layers


def time_jitter(net):
    '''
    at time t, either use sample at t-1, t, or t+1, using gather
    '''
    B_T_C = tf.shape(net)
    net = tf.reshape(net, [-1, B_T_C[-1]])

    BT = tf.shape(net)[0]
    indices = tf.range(BT, dtype=tf.int32)

    probs = [0.06, 0.88, 0.06]
    categories = tf.distributions.Categorical(probs=probs)
    # {0, 1, 2} -> {-1, 0, 1}, i.e. either use the left one, itself, or right one
    move_step = categories.sample(BT) - 1

    indices += move_step
    # edge case: if it says frame 0 replaced by frame 0 - 1, use frame 0 + 1 instead
    indices += 2 * tf.cast(indices < 0, tf.int32)
    # edge case: if it says frame -1 replaced by frame -1 + 1, use frame -1 - 1 instead
    indices -= 2 * tf.cast(indices >= BT, tf.int32)

    net = tf.gather(net, indices)
    net = tf.reshape(net, B_T_C)
    return net


def conv_3_128(net):
    return tkl.Conv1D(filters=128, kernel_size=3, dilation_rate=1, padding='same')(net)


def upsample(net, size):
    return tkl.UpSampling1D(size=size)(net)


def concat(net, global_condition):
    # global_condition = tf.expand_dims(global_condition, 1)
    global_condition = tf.tile(global_condition, [1, tf.shape(net)[1], 1])
    net = tf.concat([net, global_condition], axis=-1)
    return net



