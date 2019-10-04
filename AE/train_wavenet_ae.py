from wavenet import Wavenet
from encoder import Encoder_Magenta
import tensorflow as tf
import time, os, sys
import numpy as np
from dataset import *
# from decoder_ops import concat
from utils import display_time

if tf.__version__ == '1.14.0':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.logging.set_verbosity(tf.logging.ERROR)

dataset_args = {
    'batch_size': 8,
    'in_memory': True,
    'start': None,
    'end': None,
    'shuffle': True,
    'seed': None,
    'max_len': 5120,
    'step': None, # receptive field
    'sr': 16000
}
dataset = VCTK(**dataset_args)
num_batches = dataset.num_batches

with tf.variable_scope('encoder'):
    encoder = Encoder_Magenta()
    lc = encoder.build(net=dataset.x)

model = Wavenet()
model.build(inputs=dataset.x, local_condition=lc, global_condition=dataset.y)
model.get_loss()

save_dir = 'saved_ae_gc'
save_name = 'weights'

if len(sys.argv) < 2:
    restore_path = None
else:
    last_gs = int(sys.argv[1])
    restore_path = save_dir + '/' + save_name + '-%d'%last_gs

sess = tf.Session()
writer = tf.summary.FileWriter('logs_ae_gc', sess.graph)
saver = tf.train.Saver()
if restore_path is not None:
    saver.restore(sess, restore_path)
else:
    sess.run(tf.global_variables_initializer())

gs = sess.run(model.global_step)
lr = sess.run(model.lr)
print('last global step: %d, learning rate: %.8f' % (gs, lr))
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

num_epochs = 4
for e in range(1, num_epochs + 1):
    sess.run(dataset.init)
    step = 0
    while True:
        try:
            step += 1
            t = time.time()
            _, l, s, gs, lr = sess.run([model.opt, 
                                        model.loss, 
                                        model.summary, 
                                        model.global_step, 
                                        model.lr])
            writer.add_summary(s, gs)
            t = time.time() - t
            second = (num_batches - step) * t
            progress = '\r[epoch %d step %d] %.2f' % (e, gs, step / num_batches * 100) + '%'
            loss = ' [loss %.5f] [lr %.8f]' % (l, lr)
            print(progress + loss + display_time(t, second), end='')
        except tf.errors.OutOfRangeError:
            break
    print()
    saver.save(sess, save_dir + '/' + save_name, global_step=model.global_step)


