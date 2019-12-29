from model import VQVAE
from Encoder.encoder import *
from Decoder.decoder import *
from dataset import *
from utils import display_time, suppress_tf_warning
import tensorflow as tf
import time, os, sys, json
from argparse import ArgumentParser

suppress_tf_warning()

parser = ArgumentParser()
parser.add_argument('-dataset', default='VCTK', type=str,
                    help='VCTK or LibriSpeech or Aishell',
                    metavar='DATASET')
parser.add_argument('-length', default=6656, type=int,
                    dest='max_len', metavar='int',
                    help='number of samples one audio will contain')
parser.add_argument('-step', default=1000000, type=int,
                    dest='num_steps', metavar='int',
                    help='number of steps to train')
parser.add_argument('-batch', default=8, type=int,
                    dest='batch_size', metavar='int',
                    help='batch size')
parser.add_argument('-interval', default=200, type=int, 
                    dest='interval', metavar='int',
                    help='save log every interval step')
parser.add_argument('-restore',
                    dest='restore_path', metavar='string',
                    help='path to restore weights')
parser.add_argument('-save', default='saved_model/weights', 
                    dest='save_path', metavar='string',
                    help='path to save weights')
parser.add_argument('-params', default='model_parameters.json',
                    dest='parameter_path', metavar='str',
                    help='path to parameters file')
args = parser.parse_args()

dataset_args = {
    'relative_path': 'data/',
    'batch_size': args.batch_size,
    'max_len': args.max_len
}

if args.dataset == 'VCTK':
    dataset = VCTK(**dataset_args)
elif args.dataset == 'LibriSpeech':
    dataset = LibriSpeech(**dataset_args)
elif args.dataset == 'Aishell':
    dataset = Aishell(**dataset_args)
else:
    raise NotImplementedError('dataset %s not implemented' % args.dataset)

with open(args.parameter_path) as file:
    parameters = json.load(file)
encoders = {'Magenta': Encoder_Magenta, '64': Encoder_64, '2019': Encoder_2019}
if parameters['encoder'] in encoders:
    encoder = encoders[parameters['encoder']](parameters['latent_dim'])
else:
    raise NotImplementedError('encoder %s not implemented' % args.encoder)
decoder = WavenetDecoder(parameters['wavenet_parameters'])
model_args = {
    'x': dataset.x,
    'speaker': dataset.y,
    'encoder': encoder,
    'decoder': decoder,
    'k': parameters['k'],
    'beta': parameters['beta'],
    'verbose': parameters['verbose'],
    'use_vq': parameters['use_vq'],
    'speaker_embedding': parameters['speaker_embedding'],
    'num_speakers': dataset.num_speakers
}

schedule = {int(k): v for k, v in parameters['learning_rate_schedule'].items()}

model = VQVAE(model_args)
model.build(learning_rate_schedule=schedule)

sess = tf.Session()
saver = tf.train.Saver()

if args.restore_path is not None:
    saver.restore(sess, args.restore_path)
else:
    sess.run(tf.global_variables_initializer())

gs = sess.run(model.global_step)
lr = sess.run(model.lr)
print('[restore] last global step: %d, learning rate: %.5f' % (gs, lr))

save_path = args.save_path
save_dir, save_name = save_path.split('/')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

writer = tf.summary.FileWriter(save_dir, sess.graph)

sess.run(dataset.init)
for step in range(1, 1 + args.num_steps): 
    try:
        t = time.time()
        if (gs + 1) % args.interval == 0:
            _, rl, gs, lr, summary = sess.run([model.train_op, 
                                               model.reconstruction_loss, 
                                               model.global_step, 
                                               model.lr,
                                               model.summary])
            writer.add_summary(summary, gs)
        else:
            _, rl, gs, lr = sess.run([model.train_op, 
                                               model.reconstruction_loss, 
                                               model.global_step, 
                                               model.lr])
        t = time.time() - t
        progress = '\r[step %d] %.2f' % (gs, step / args.num_steps * 100) + '%'
        loss = ' [recons %.5f] [lr %.5f]' % (rl, lr)
        second = (args.num_steps - step) * t
        print(progress + loss + display_time(t, second), end='')
    except tf.errors.OutOfRangeError:
        print('\ndataset re-initialised')
        sess.run(dataset.init)
saver.save(sess, save_path, global_step=model.global_step)

