from model import VQVAE
from Encoder.encoder import *
from Decoder.decoder import *
from utils import get_speaker_to_int, decode, suppress_tf_warning
import tensorflow as tf
import numpy as np
import time, json
from tqdm import tqdm
from scipy.io import wavfile
from argparse import ArgumentParser

suppress_tf_warning()

parser = ArgumentParser()
parser.add_argument('-restore',
                    dest='restore_path',
                    help='path to weights')
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
parser.add_argument('-params', default='model_parameters.json',
                    dest='parameter_path', metavar='str',
                    help='path to parameters file')

args = parser.parse_args()
gs = int(args.restore_path.split('-')[-1])
batch_size = len(args.speakers)

content = tf.io.read_file(args.audio_path)
wav = tf.contrib.ffmpeg.decode_audio(content, 'wav', 16000, 1)
# 512 is the largest dilation rate; change as needed
wav = tf.reshape(wav[:tf.shape(wav)[0] // 512 * 512, :], [1, -1, 1])
wav = tf.tile(wav, [batch_size, 1, 1])

sess = tf.Session()
wav = sess.run(wav)
length = wav.shape[1]

if args.speakers[0][0] == 'p': # VCTK
    speaker_to_int = get_speaker_to_int('data/vctk_speakers.txt')
    speaker = [[0 for _ in range(109)] for _ in range(len(args.speakers))]
    num_speakers = 109
elif args.speakers[0][0].lower() == 's': # aishell
    speaker_to_int = get_speaker_to_int('data/aishell_speakers.txt')
    speaker = [[0 for _ in range(340)] for _ in range(len(args.speakers))]
    num_speakers = 340
else: # LibriSpeech
    speaker_to_int = get_speaker_to_int('data/librispeech_speakers.txt')
    speaker = [[0 for _ in range(251)] for _ in range(len(args.speakers))]
    num_speakers = 251
for i, s in enumerate(args.speakers):
    if s.lower() != 'None':
        speaker[i][speaker_to_int[s]] = 1
speaker = np.expand_dims(speaker, 1)

with open(args.parameter_path) as file:
    parameters = json.load(file)
encoders = {'Magenta': Encoder_Magenta, '64': Encoder_64, '2019': Encoder_2019}
if parameters['encoder'] in encoders:
    encoder = encoders[parameters['encoder']](parameters['latent_dim'])
else:
    raise NotImplementedError("encoder %s not implemented" % args.encoder)
decoder = WavenetDecoder(parameters['wavenet_parameters'])
model_args = {
    'x': wav,
    'speaker': tf.constant(speaker, dtype=np.float32),
    'encoder': encoder,
    'decoder': decoder,
    'k': parameters['k'],
    'beta': parameters['beta'],
    'verbose': parameters['verbose'],
    'use_vq': parameters['use_vq'],
    'speaker_embedding': parameters['speaker_embedding'],
    'num_speakers': num_speakers
}

model = VQVAE(model_args)
model.build_generator()
wavenet = model.decoder.wavenet

variables = model.ema.variables_to_restore()
saver = tf.train.Saver(variables)
saver.restore(sess, args.restore_path)

encoding = sess.run(model.encoding)

save_path = args.restore_path.split('/weights')[0]

if parameters['use_vq']:
    embedding = sess.run(model.embedding)
    np.save(save_path + '/embedding_%d.npy' % gs, embedding)
if parameters['speaker_embedding'] > 0:
    embedding = sess.run(model.speaker_embedding)
    np.save(save_path + '/speaker_embedding_%d.npy' % gs, embedding)

audio = np.zeros([batch_size, 1], dtype=np.float32)
to_write = np.zeros([batch_size, length], dtype=np.float32)
sess.run(wavenet.init_ops)

ratio = length // encoding.shape[1]
for i in tqdm(range(length)):
    probs, _ = sess.run([wavenet.predictions, wavenet.push_ops], 
        {wavenet.input_t: audio, wavenet.local_condition_t: encoding[:, i // ratio]})
    decoded = decode(probs, mode=args.mode, quantization_channels=wavenet.args['quantization_channels'])
    to_write[:, i] = decoded
    audio = np.expand_dims(decoded, -1)

for i, s in enumerate(args.speakers):
    s = 'no_speaker' if s == 'None' else s
    wavfile.write(save_path + '/%d_%s.wav'%(gs, s), 16000, to_write[i])

