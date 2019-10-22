import sys
sys.path.append("..")
from config import Config, FastGenerationConfig
from mu_law_ops import *
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np
from utils import get_speaker_to_int, decode
from argparse import ArgumentParser

if tf.__version__ == '1.14.0':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.logging.set_verbosity(tf.logging.ERROR)
    
parser = ArgumentParser()
parser.add_argument('-restore',
                    dest='restore_path',
                    help='path to weights')
parser.add_argument('-save',
                    dest='save_name',
                    help='filename to save as')
parser.add_argument('-audio',
                    dest='audio_path',
                    help='path to audio')
parser.add_argument('-speakers', 
                    nargs='+',
                    dest='speakers',
                    help='speaker id')
parser.add_argument('-mode', default='sample', 
                    dest='mode',
                    help='decode mode, sample or greedy')

args = parser.parse_args()
gs = int(args.restore_path.split('-')[-1])
batch_size = len(args.speakers)

content = tf.io.read_file(args.audio_path)
wav = tf.contrib.ffmpeg.decode_audio(content, 'wav', 16000, 1)
wav = tf.reshape(wav[:tf.shape(wav)[0] // 512 * 512, :], [1, -1, 1])
wav = tf.tile(wav, [batch_size, 1, 1])
sess = tf.Session()  
wav = tf.constant(sess.run(wav))

speaker_to_int = get_speaker_to_int('../data/vctk_speakers.txt')
speaker = [[0] * 109] * len(args.speakers)
for i, s in enumerate(args.speakers):
    speaker[i][speaker_to_int[s]] = 1
speaker = np.asarray(speaker, dtype=np.float32)
gc = tf.constant(speaker)

config = Config()
config.build(inputs=wav, gc=None, is_training=False)

generator = FastGenerationConfig(batch_size=batch_size)
x_t = tf.placeholder(tf.float32, shape=[batch_size, 1])
net = generator.build(inputs=x_t, gc=gc)

saver = tf.train.Saver(generator.shadow)
saver.restore(sess, args.restore_path)

encoding = sess.run(config.encoding)
length = wav.get_shape().as_list()[1]
print('embedding:')
emb = sess.run(config.embedding)
print(emb)
print('running encoding:', encoding.shape)

audio = np.zeros([batch_size, 1], dtype=np.float32)
to_write = np.zeros([batch_size, length], dtype=np.float32)
sess.run(net['init_ops'])

ratio = length // encoding.shape[1]
for i in tqdm(range(length)):
    probs, _ = sess.run([net['predictions'], net['push_ops']], 
        {x_t: audio, net['encoding']: encoding[:, i // ratio]})
    decoded = decode(probs, mode=args.mode)
    to_write[:, i] = decoded
    audio = decoded.reshape([batch_size, 1])

for i, s in enumerate(args.speakers):
    wavfile.write(args.save_name + '_%d_%s.wav'%(gs, s), 16000, to_write[i])

