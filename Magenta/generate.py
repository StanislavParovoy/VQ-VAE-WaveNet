import sys
sys.path.append("..")
from masked import *
from config import Config, FastGenerationConfig
from mu_law_ops import *
from tqdm import tqdm
import numpy as np
from utils import get_speaker_to_int

if tf.__version__ == '1.14.0':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.logging.set_verbosity(tf.logging.ERROR)

tf.reset_default_graph()  

path = ''
name = '../p225_001.wav'
content = tf.io.read_file(path + name)
# wav = tf.audio.decode_wav(content, desired_channels=1)
wav = tf.contrib.ffmpeg.decode_audio(content, 'wav', 16000, 1)
ratio = 64
wav = tf.reshape(wav[:tf.shape(wav)[0] // ratio * ratio, :], [1, -1, 1])
# sample_length = tf.shape(wav)[1]
batch_size = 1

person = sys.argv[2]
speaker_to_int = get_speaker_to_int('../vctk_speakers.txt')
speaker = [[0] * 109]
speaker[0][speaker_to_int[person]] = 1
speaker = np.asarray(speaker, dtype=np.float32)
gc = tf.constant(speaker)

config = Config()
# x = tf.placeholder(tf.float32, shape=[batch_size, sample_length, 1])
config.build(inputs=wav, gc=None, is_training=False)

generator = FastGenerationConfig(batch_size=batch_size)
x_t = tf.placeholder(tf.float32, shape=[batch_size, 1])
net = generator.build(inputs=x_t, gc=gc)
net['x_t'] = x_t

def sample(probs):
  cdf = np.cumsum(probs, axis=1)
  cdf = cdf.reshape([-1])
  pred = cdf.searchsorted(np.random.rand())
  # pred = np.argmax(net['predictions'], axis=-1)
  raw = np.reshape(pred, [])
  dec = np.reshape(mu_law_decode_np(pred), [])
  return raw, dec

gs = int(sys.argv[1])
saved_path = 'saved_vqvae_config/weights-%d'%gs
sess = tf.Session()  
saver = tf.train.Saver(generator.shadow)
saver.restore(sess, saved_path)

encoding = sess.run(config.encoding)
length = sess.run(wav).shape[1]
print('running encoding:', encoding.shape)
print('running gc:', speaker.shape)
# sess.run(tf.global_variables_initializer())
audio = np.zeros([batch_size, 1], dtype=np.float32)
total = []
sess.run(net['init_ops'])
for i in tqdm(range(length)):
  probs, _ = sess.run([net['predictions'], net['push_ops']], {x_t: audio, net['encoding']: encoding[:, i // ratio]})
  raw, dec = sample(probs)
  total.append(dec.reshape([]))
  audio = dec.reshape([batch_size, 1])
from scipy.io import wavfile
wavfile.write('vqvae_config_%d_%s.wav'%(gs, person), 16000, np.asarray(total))


