from config import Config
import tensorflow as tf
import time, os, sys
import numpy as np
from dataset import *
from utils import display_time

if tf.__version__ == '1.14.0':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.logging.set_verbosity(tf.logging.ERROR)

dataset_args = {
    'batch_size': 8,
    'in_memory': False,
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

model = Config()
model.build(inputs=dataset.x, gc=dataset.y, is_training=True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('logs_vqvae_config', sess.graph)
saver = tf.train.Saver()

save_dir = 'saved_vqvae_config'
save_name = 'weights'

if len(sys.argv) < 2:
    restore_path = None
else:
    last_gs = int(sys.argv[1])
    restore_path = save_dir + '/' + save_name + '-%d'%last_gs

if restore_path is not None:
    saver.restore(sess, restore_path)
else:
    sess.run(tf.global_variables_initializer())

gs = sess.run(model.global_step)
lr = sess.run(model.lr)
print('last global step: %d, learning rate: %.8f' % (gs, lr))
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
    
sess.run(dataset.init)
step = 0
num_epochs = 3
for e in range(num_epochs):
    while True:
        try:
            step += 1
            t = time.time()
            _, l, summary, gs, lr = sess.run([model.opt, 
                                              model.loss, 
                                              model.summary, 
                                              model.global_step,
                                              model.lr])
            writer.add_summary(summary, gs)
            t = time.time() - t
            progress = '\r[e %d step %d] %.2f' % (e, gs, step / num_batches * 100) + '%'
            loss = ' [loss %.5f] [lr %.7f]' % (l, lr)
            second = (num_batches - step) * t
            print(progress + loss + display_time(t, second), end='')
        except tf.errors.OutOfRangeError:
            break
    print()
    saver.save(sess, save_dir + '/' + save_name, global_step=model.global_step)
