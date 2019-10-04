from model import VQVAE
from Encoder.encoder import *
from Decoder.decoder import *
from utils import decode
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import time
from utils import get_speaker_to_int

if tf.__version__ == '1.14.0':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.logging.set_verbosity(tf.logging.ERROR)

ratio = 512
# path = 'VCTK-Corpus/wav48/p225/'
path = ''
name = 'p225_001.wav'
content = tf.io.read_file(path + name)
# wav = tf.audio.decode_wav(content, desired_channels=1)
wav = tf.contrib.ffmpeg.decode_audio(content, 'wav', 16000, 1)
wav = tf.reshape(wav[:tf.shape(wav)[0] // ratio * ratio, :], [1, -1, 1])

sess = tf.Session()
wav = sess.run(wav)
length = wav.shape[1]

speaker_to_int = get_speaker_to_int('vctk_speakers.txt')
speaker = [[0] * 109]
speaker[0][speaker_to_int['p225']] = 1
speaker = np.asarray(speaker, dtype=np.float32)

wavenet_args_path = 'wavenet.json'
encoder = Encoder_Magenta()
decoder = WavenetDecoder(wavenet_args_path)
model_args = {
    'x': tf.constant(wav),
    'speaker': tf.constant(speaker),
    'encoder': encoder,
    'decoder': decoder,
    'latent_dim': 64,
    'k': 512, 
    'beta': 0.25,
    'verbose': True
}

model = VQVAE(model_args)
model.build_generator()
wavenet = model.decoder.wavenet

import sys
gs = int(sys.argv[1])
#gs = 12004
saved_path = 'saved_vqvae/weights-%d'%gs
saver = tf.train.Saver(var_list=tf.trainable_variables())
saver.restore(sess, saved_path)

encoding = sess.run(model.z_q)
print('numpy encoding:', encoding.shape)

decode_mode = 'sample'

batch_size = 1
audio = np.zeros([batch_size, 1], dtype=np.float32)
total = []
sess.run(wavenet.init_ops)
for i in tqdm(range(length)):
    probs, _ = sess.run([wavenet.predictions, wavenet.push_ops], 
        {wavenet.input_t: audio, wavenet.local_condition_t: encoding[:, i // ratio]})
    raw, decoded = decode(probs, decode_mode)
    total.append(decoded.reshape([]))
    audio = decoded.reshape([batch_size, 1])

wavfile.write('generated_vqvae.wav', 16000, np.asarray(total))











