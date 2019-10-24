import sys
sys.path.append("..")
from config import Config
import tensorflow as tf
import time, os
import numpy as np
from dataset import *
from utils import display_time
from argparse import ArgumentParser

if tf.__version__ == '1.14.0':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.logging.set_verbosity(tf.logging.ERROR)

parser = ArgumentParser()
parser.add_argument('dataset',
                    help='VCTK or LibriSpeech',
                    metavar='DATASET')
parser.add_argument('-m', default=1, type=int,
                    dest='in_memory', metavar='bool',
                    help='if loading data in memory')
parser.add_argument('-l', default=6144, type=int,
                    dest='max_len', metavar='int',
                    help='number of samples one audio will contain')
parser.add_argument('-e', default=1, type=int,
                    dest='num_epochs', metavar='int',
                    help='number of epoch to train')
parser.add_argument('-b', default=4, type=int,
                    dest='batch_size', metavar='int',
                    help='batch size')
parser.add_argument('-log', default='logs',
                    dest='log_path', metavar='string',
                    help='path to save logs for tensorboard')
parser.add_argument('-restore',
                    dest='restore_path', metavar='string',
                    help='path to restore weights')
parser.add_argument('-save', default='saved_model/weights',
                    dest='save_path', metavar='string',
                    help='path to save weights')
args = parser.parse_args()

dataset_args = {
    'relative_path': '../data/',
    'batch_size': args.batch_size,
    'in_memory': args.in_memory,
    'max_len': args.max_len,
    'sr': 16000
}

if args.dataset == 'VCTK':
    dataset = VCTK(**dataset_args)
elif args.dataset == 'LibriSpeech':
    dataset = LibriSpeech(**dataset_args)
num_batches = dataset.num_batches

model = Config()
model.build(inputs=dataset.x, gc=dataset.y, is_training=True)
sess = tf.Session()

writer = tf.summary.FileWriter(args.log_path, sess.graph)
# saver = tf.train.Saver(model.variables)
saver = tf.train.Saver()

if args.restore_path is not None:
    saver.restore(sess, args.restore_path)
else:
    sess.run(tf.global_variables_initializer())

gs = sess.run(model.global_step)
lr = sess.run(model.lr)
print('last global step: %d, learning rate: %.5f' % (gs, lr))

save_path = args.save_path
save_dir, save_name = save_path.split('/')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
    
for e in range(args.num_epochs):
    sess.run(dataset.init)
    step = 0
    while True:
        try:
            step += 1
            t = time.time()
            _, l, summary, gs, lr = sess.run([model.opt, 
                                              model.reconstruction_loss, 
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
    saver.save(sess, save_path, global_step=model.global_step)

