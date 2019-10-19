import tensorflow as tf
from Decoder.WaveNet.wavenet_ops import *
from mu_law_ops import *
import json


class Wavenet():


    def __init__(self, args_file='wavenet_parameters.json'):
        with open(args_file) as file:
            args = json.load(file)
        assert len(args['dilation_rates']) == args['num_cycles'] * args['num_cycle_layers']

        kernel_size = args['kernel_size']
        self.receptive_field = sum(args['dilation_rates']) * (kernel_size - 1) + 1
        print('receptive_field:', self.receptive_field)
        self.receptive_field += args['preprocess']['kernel_size'] - 1
        print('receptive_field after preprocess:', self.receptive_field)

        self.args = args


    def build(self, inputs, local_condition=None, global_condition=None):
        ''' the wavenet decoder
        args:
            inputs: original raw audio, shape [b, t, 1]
            local_condition: output from VQVAE, upsampled, shape [b, t, 128 + num_speaker]
        returns:
            wavenet output, same shape as inputs
        '''

        inputs = mu_law_encode(inputs)
        print('wavenet inputs:', inputs.shape)
        
        # [-1, 1] -> [0, 255]
        mu = self.args['quantization_channels'] - 1
        labels = tf.cast((inputs + 1) / 2 * mu + 0.5, tf.int32)
        self.labels = tf.reshape(labels, [-1])

        inputs = shift_right(inputs)

        # preprocess layer
        with tf.variable_scope('preprocess'):
            args = self.args['preprocess']
            net = conv1d_v2(inputs, args['filters'], args['kernel_size'])
            print('net preprocess:', net.shape)

        if local_condition is not None:
            print('local_condition:', local_condition.shape)
        if global_condition is not None:
            print('global_condition:', global_condition.shape)

        # skip starts from preprocess
        with tf.variable_scope('skip'):
            skip = conv1d_v2(net, self.args['skip_filters'], kernel_size=1)
            print('skip start:', skip.shape)

        # residual stacks
        kernel_size = self.args['kernel_size']
        dilation_filters = self.args['dilation_filters']
        skip_filters = self.args['skip_filters']
        residual_filters = self.args['residual_filters']

        for i, dilations in enumerate(self.args['dilation_rates']):
            cycle_id = 'cycle_%d' % (1 + i // self.args['num_cycle_layers'])
            layer_id = 'layer_%d' % (1 + i % self.args['num_cycle_layers'])
            with tf.variable_scope(cycle_id + '/' + layer_id):
                print('dr_%d: res_out: ' % dilations, end='')

                skip_out, res_out = residual_stack(net, \
                    dilation_filters, kernel_size, dilations, \
                    skip_filters, residual_filters, \
                    local_condition, global_condition)

                skip += skip_out
                net += res_out

        net = skip
        print('skip sum:', net.shape)

        # postprocess layer 1 with condition
        with tf.variable_scope('postprocess1'):
            net = tf.nn.relu(net)
            net = conv1d_v2(net, self.args['skip_filters'], kernel_size=1)
            print('net postprocess 1:', net.shape)

            # add condition
            if local_condition is not None:
                with tf.variable_scope('local_condition'):
                    net = add_condition(net, local_condition)
                with tf.variable_scope('global_condition'):
                    net = add_condition(net, global_condition)
                print('net postprocess 1 condition:', net.shape)

        # postprocess layer 2, outputs logits
        with tf.variable_scope('postprocess2'):
            net = tf.nn.relu(net)
            net = conv1d_v2(net, self.args['quantization_channels'], kernel_size=1)
            print('net postprocess 2:', net.shape)

        self.logits = tf.reshape(net, [-1, self.args['quantization_channels']])
        return self.logits, self.labels


    def build_generator(self, input_t, local_condition_t, global_condition_t,
        batch_size=1, verbose=False):
        ''' performs the fast wavenet generation for one step
        args:
            input_t: initial value (at first time stamp)
            condition_t: initial value of condition_t (at first time stamp)
        returns:
            value at final layer of wavenet (at second time stamp)
        '''
        self.input_t = input_t
        input_t = mu_law_encode(input_t)
        init_ops = []
        push_ops = []

        state_size = input_t.shape.as_list()[-1]
        q = tf.FIFOQueue(1,
                         dtypes=tf.float32,
                         shapes=(batch_size, state_size))
        init = q.enqueue_many(tf.zeros((1, batch_size, state_size)))

        past = q.dequeue()
        push = q.enqueue([input_t])
        init_ops.append(init)
        push_ops.append(push)

        # preprocess layer
        with tf.variable_scope('preprocess'):
            args = self.args['preprocess']
            current = fast_conv1d(past, input_t, args['filters'], args['kernel_size'])
            if verbose:
                print('net preprocess:', current.shape)

        # skip starts from preprocess
        with tf.variable_scope('skip'):
            skip = linear(current, filters=self.args['skip_filters'])

        # residual stacks
        kernel_size = self.args['kernel_size']
        state_size = self.args['dilation_filters']
        skip_filters = self.args['skip_filters']
        residual_filters = self.args['residual_filters']

        for i, dilations in enumerate(self.args['dilation_rates']):
            cycle_id = 'cycle_%d' % (1 + i // self.args['num_cycle_layers'])
            layer_id = 'layer_%d' % (1 + i % self.args['num_cycle_layers'])
            with tf.variable_scope(cycle_id + '/' + layer_id):
                q = tf.FIFOQueue(dilations,
                                 dtypes=tf.float32,
                                 shapes=(batch_size, state_size))
                init = q.enqueue_many(tf.zeros((dilations, batch_size, state_size)))

                # dequeue past, enqueue current
                past = q.dequeue()
                push = q.enqueue([current])
                init_ops.append(init)
                push_ops.append(push)

                skip_out, res_out = fast_residual_stack(past, current, state_size, \
                    kernel_size, local_condition_t, global_condition_t, \
                    skip_filters, residual_filters)
                if verbose:
                    print('skip:', skip_out.shape, '   res:', res_out.shape)
                skip += skip_out
                current += res_out

        net = skip
        if verbose:
            print('sum skip:', net.shape)

        # postprocess layer 1 with condition
        with tf.variable_scope('postprocess1'):
            net = tf.nn.relu(net)
            net = linear(net, self.args['skip_filters'])
            if verbose:
                print('net postprocess 1:', net.shape)

            # # add condition
            if local_condition_t is not None:
                with tf.variable_scope('local_condition'):
                    net = fast_condition(net, local_condition_t)
                with tf.variable_scope('global_condition'):
                    net = fast_condition(net, global_condition_t)
                if verbose:
                    print('net postprocess 1 condition:', net.shape)

        # # postprocess layer 2, outputs logits
        with tf.variable_scope('postprocess2'):
            net = tf.nn.relu(net)
            net = linear(net, self.args['quantization_channels'])
            if verbose:
                print('net postprocess 2:', net.shape)

        self.init_ops = init_ops
        self.push_ops = push_ops
        self.predictions = tf.nn.softmax(net)
        self.local_condition_t = local_condition_t
        return net


    def get_loss(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=self.labels,
                                    logits=self.logits)
        self.loss = tf.reduce_mean(loss, axis=0)

        self.global_step = tf.Variable(tf.constant(0, dtype=tf.int32), trainable=False)

        # self.lr = tf.train.exponential_decay(
        #                     learning_rate=tf.Variable(0.0001),
        #                     global_step=self.global_step,
        #                     decay_steps=10000,
        #                     decay_rate=0.96)
        learning_rate_schedule = {
            0: 0.0001,
            30000: 0.00008,
            40000: 0.00006,
            60000: 0.00004,
            80000: 0.00002,
            100000: 0.00001
        }
        self.lr = tf.constant(learning_rate_schedule[0])
        for key, value in learning_rate_schedule.iteritems():
            self.lr = tf.cond(
                tf.less(self.global_step, key), lambda: self.lr, lambda: tf.constant(value))
        optimiser = tf.train.AdamOptimizer(self.lr)
        self.opt = optimiser.minimize(self.loss, global_step=self.global_step)
        tf.summary.scalar('loss', self.loss)
        self.summary = tf.summary.merge_all()

