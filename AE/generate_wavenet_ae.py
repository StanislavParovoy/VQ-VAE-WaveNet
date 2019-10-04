from encoder import Encoder_Magenta
from wavenet import Wavenet
from decoder_ops import concat
from mu_law_ops import *
from utils import get_speaker_to_int
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile


if tf.__version__ == '1.14.0':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.logging.set_verbosity(tf.logging.ERROR)

tf.reset_default_graph()  

path = 'VCTK-Corpus/wav48/p225/'
path = ''
name = 'p225_094.wav'
name = 'p225_001.wav'
# name = 'c.wav'
content = tf.io.read_file(path + name)
# wav = tf.audio.decode_wav(content, desired_channels=1)
wav = tf.contrib.ffmpeg.decode_audio(content, 'wav', 16000, 1)
ratio = 512
wav = tf.reshape(wav[:tf.shape(wav)[0] // ratio * ratio, :], [1, -1, 1])

sess = tf.Session()
wav = sess.run(wav)
print('wav:', wav.shape)
length = wav.shape[1]

speaker_to_int = get_speaker_to_int('vctk_speakers.txt')
speaker = [[0] * 109]
speaker[0][speaker_to_int['p225']] = 1
# speaker[0][3] = 1
speaker = np.asarray(speaker, dtype=np.float32)

batch_size = 1

with tf.variable_scope('encoder'):
    encoder = Encoder_Magenta()
    encoding = encoder.build(net=tf.constant(wav))
    # encoding = concat(lc, global_condition=tf.constant(speaker))
    print('encoding:', encoding.shape)

input_t = tf.placeholder(tf.float32, [None, 1])
channels = encoding.shape.as_list()[-1]
local_condition_t = tf.placeholder(tf.float32, [None, channels])
global_condition_t = tf.constant(speaker)
model = Wavenet()
model.build_generator(input_t, local_condition_t, global_condition_t, 1)

def sample(probs):
    cdf = np.cumsum(probs, axis=1)
    cdf = cdf.reshape([-1])
    pred = cdf.searchsorted(np.random.rand())
    # pred = np.argmax(net['predictions'], axis=-1)
    raw = np.reshape(pred, [])
    dec = np.reshape(mu_law_decode_np(pred), [])
    return raw, dec

import sys
gs = int(sys.argv[1])
#gs = 12004
saved_path = 'saved_ae_gc/weights-%d'%gs
saver = tf.train.Saver(var_list=tf.trainable_variables())
saver.restore(sess, saved_path)

encoding = sess.run(encoding)
print('running encoding:', encoding.shape)
audio = np.zeros([batch_size, 1], dtype=np.float32)
total = []
sess.run(model.init_ops)
for i in tqdm(range(length)):
  probs, _ = sess.run([model.predictions, model.push_ops], {input_t: audio, local_condition_t: encoding[:, i // ratio]})
  raw, dec = sample(probs)
  total.append(dec.reshape([]))
  audio = dec.reshape([batch_size, 1])

wavfile.write('generated_ae_gc.wav', 16000, np.asarray(total))

