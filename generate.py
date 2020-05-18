import tensorflow as tf
import numpy as np
import time, json, librosa
from vqvae import VQVAE
from Encoder.encoder import *
from Decoder.WaveNet.wavenet import *
from utils import get_speaker_to_int, decode, suppress_tf_warning
import parameters
from tqdm import tqdm
from scipy.io import wavfile
from argparse import ArgumentParser

def parse_args():
  parser = ArgumentParser()
  parser.add_argument('-restore', help='path to weights')
  parser.add_argument('-audio', help='path to audio')
  parser.add_argument('-gc', default='accent', choices=['speaker', 'accent'])
  parser.add_argument('-speakers', nargs='+')
  args = parser.parse_args()
  return args

def gaussian_sample(logits):
  u, log_s = logits[:, 0], logits[:, 1]
  log_s = logits[:, 1]
  # log_s = np.maximum(logits[:, 1], -7.)
  s = np.exp(log_s)
  b = logits.shape[0]
  samples = np.random.normal(loc=u, scale=s, size=b)
  samples = np.maximum(np.minimum(samples, 1), -1)

  q = (2**16) / 2
  samples = np.floor(samples * q).astype(np.int32)
  samples = samples.astype(np.float32) / q
  return samples  

def main():
  args = parse_args()
  if args.gc == 'speaker':
    batch_size = len(args.speakers)
  else:
    batch_size = 11

  wav, _ = librosa.load(args.audio, sr=parameters.sr)
  wav = np.reshape(wav[: wav.shape[0] // 64 * 64], [1, -1, 1])
  wav = np.reshape(wav, [1, -1, 1])
  wav = np.tile(wav, [batch_size, 1, 1])
  length = wav.shape[1]
  print('length:', wav.shape)

  if args.gc == 'speaker':
    speaker_to_int = get_speaker_to_int('vctk_speakers.txt')
    # speaker_to_int = get_speaker_to_int('data/vctk_info/vctk_speakers.txt')
    h = [[speaker_to_int[s]] for s in args.speakers]
    gc_dim = 109
  else:
    h = [[i] for i in range(batch_size)]
    gc_dim = batch_size
  print('gc:', np.shape(h))

  encoder = Encoder(scope='encoder', latent_dim=parameters.latent_dim, trainable=False)
  vqvae = VQVAE(scope='vqvae', trainable=False)
  condition = vqvae(encoder(wav))
  print('lc:', condition.shape)

  gc_embedding = tf.get_variable(
      name='decoder/gc_embedding', 
      shape=[gc_dim, parameters.global_embedding_dim])
  h = tf.nn.embedding_lookup(gc_embedding, h)
  h = tf.tile(h, [1, condition.get_shape().as_list()[1], 1])
  print('gc:', h.shape)
  condition = tf.concat([condition, h], axis=-1)
  print('condition:', condition.shape)

  decoder = WaveNet(args=parameters.teacher, scope='decoder', trainable=False)
  input_t = tf.placeholder(tf.float32, [batch_size, 1])
  condition_t = tf.placeholder(tf.float32, [batch_size, condition.shape[-1]])
  init_ops, push_ops, x = decoder.generate(input_t, condition_t, batch_size=batch_size)
  exit()
  sess = tf.Session()
  saver = tf.train.Saver()
  latest_checkpoint = tf.train.latest_checkpoint(args.restore)
  gs = int(latest_checkpoint.split('-')[-1])
  saver.restore(sess, latest_checkpoint)

  encoding = sess.run(condition)

  if parameters.use_vq:
    embedding = sess.run(vqvae.codebook)
    np.save(args.restore + '/embedding_%d.npy' % gs, embedding)
  if parameters.global_embedding_dim > 0:
    embedding = sess.run(gc_embedding)
    np.save(args.restore + '/gc_embedding_%d.npy' % gs, embedding)

  audio = np.zeros([batch_size, 1], dtype=np.float32)
  to_write = np.zeros([batch_size, length], dtype=np.float32)
  sess.run(init_ops)

  ratio = length // encoding.shape[1]
  print('ratio:', ratio)
  for i in tqdm(range(length)):
    probs, _ = sess.run([x, push_ops], 
        {input_t: audio, condition_t: encoding[:, i // ratio]})
    # decoded = decode(probs, mode=args.mode, quantization_channels=wavenet.args['quantization_channels'])
    decoded = gaussian_sample(probs)
    to_write[:, i] = decoded
    audio = np.expand_dims(decoded, -1)
  
  if args.gc == 'speaker':
    for i, s in enumerate(args.speakers):
      s = 'no_speaker' if s == 'None' else s
      wavfile.write(args.restore + '/%d_%s.wav'%(gs, s), parameters.sr, to_write[i])
  else:
    for i in range(batch_size):
      wavfile.write(args.restore + '/%d_accent_%s.wav'%(gs, i), parameters.sr, to_write[i])

if __name__ == '__main__':
  suppress_tf_warning()
  main()


