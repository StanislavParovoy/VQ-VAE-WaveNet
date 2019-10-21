from model import VQVAE
from Encoder.encoder import *
from Decoder.decoder import *
from utils import get_speaker_to_int, decode
import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm
from scipy.io import wavfile

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
parser.add_argument('-ratio', default=64, type=int, 
                    dest='ratio',
                    help='local condition ratio')
parser.add_argument('-mode', default='sample', 
                    dest='mode',
                    help='decode mode, sample or greedy')

args = parser.parse_args()
gs = int(args.restore_path.split('-')[-1])
batch_size = len(args.speakers)

content = tf.io.read_file(args.audio_path)
wav = tf.contrib.ffmpeg.decode_audio(content, 'wav', 16000, 1)
wav = tf.reshape(wav[:tf.shape(wav)[0] // args.ratio * args.ratio, :], [1, -1, 1])
wav = tf.tile(wav, [batch_size, 1, 1])

speaker_to_int = get_speaker_to_int('../data/vctk_speakers.txt')
speaker = [[0] * 109] * len(args.speakers)
for i, s in enumerate(args.speakers):
    speaker[i][speaker_to_int[s]] = 1
speaker = np.asarray(speaker, dtype=np.float32)

with open(args.parameter_path) as file:
    parameters = json.load(file)
encoders = {'Magenta': Encoder_Magenta, '64': Encoder_64, '2019': Encoder_2019}
if args.encoder in encoders:
    encoder = encoders[args.encoder](parameters['latent_dim'])
else:
    raise NotImplementedError("encoder %s not implemented" % args.encoder)
decoder = WavenetDecoder(parameters['wavenet_parameters'])
model_args = {
    'x': wav,
    'speaker': tf.constant(speaker),
    'encoder': encoder,
    'decoder': decoder,
    'k': parameters['k'],
    'beta': parameters['beta'],
    'verbose': parameters['verbose']
}

model = VQVAE(model_args)
model.build_generator(batch_size=batch_size)
wavenet = model.decoder.wavenet

sess = tf.Session()  
saver = tf.train.Saver()
saver.restore(sess, args.restore_path)

encoding = sess.run(model.z_q)
print('running encoding:', encoding.shape)
length = sess.run(wav).shape[1]
print('embedding:')
emb = sess.run(model.embedding)
print(emb)

audio = np.zeros([batch_size, 1], dtype=np.float32)
to_write = np.zeros([batch_size, length], dtype=np.float32)
sess.run(wavenet.init_ops)
for i in tqdm(range(length)):
    probs, _ = sess.run([wavenet.predictions, wavenet.push_ops], 
        {wavenet.input_t: audio, wavenet.local_condition_t: encoding[:, i // args.ratio]})
    decoded = decode(probs, mode=args.mode)
    to_write[:, i] = decoded
    audio = decoded.reshape([batch_size, 1])

for i, s in enumerate(args.speakers):
    wavfile.write(args.save_name + '_%d_%s.wav'%(gs, s), 16000, to_write[i])

