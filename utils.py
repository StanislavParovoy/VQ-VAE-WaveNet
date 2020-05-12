from mu_law_ops import mu_law_decode_np
import os
import numpy as np
import tensorflow as tf

def suppress_tf_warning():
  if tf.__version__ in {'1.14.0', '1.15.0'}:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  else:
    tf.logging.set_verbosity(tf.logging.ERROR)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def sample(pdf, quantization_channels=256):
  ''' sample from pdf
  args:
    pdf: pdf, shape [b, quantization_channels]
  returns:
    sampled and mu_law decoded, shape [b], in range [-1, 1]
  '''
  cdf = np.cumsum(pdf, axis=1)
  batch_size = cdf.shape[0]
  sample_prob = np.random.rand(batch_size)
  pred = np.zeros(batch_size, dtype=np.float32)
  for i, prob in enumerate(sample_prob):
    pred[i] = cdf[i].searchsorted(prob)
  decoded = mu_law_decode_np(pred, quantization_channels=quantization_channels)
  return decoded

def decode(predictions, mode='sample', quantization_channels=256):
  ''' decode from raw output
  args:
    mode: either sample or greedy
  returns:
    a tuple of:
      decoded of shape [b] in range [-1, 1], this is for writing audio file
      pred of shape [b] in range [0, quantization_channels], 
        this is for feeding next audio input for wavenet
  '''
  if mode == 'sample':
    return sample(predictions)
  elif mode == 'greedy':
    pred = np.argmax(predictions, axis=-1)
    return mu_law_decode_np(pred, quantization_channels=quantization_channels)
  else:
    raise NotImplementedError("decode mode %s not implemented" % mode)

def write_speaker_to_int(dataset='vctk'):
  if dataset == 'vctk':
    func = lambda s: s.split('/')[0]
    file_list = 'vctk_train.txt'
    write_as = 'vctk_speakers.txt'
  elif dataset == 'librispeech':
    func = lambda s: s.split('/')[-1].split('-', 1)[0]
    file_list = 'librispeech_train_clean_100.txt'
    write_as = 'librispeech_speakers.txt'
  else:
    raise NotImplementedError("dataset %s not implemented" % dataset)

  speaker_to_int = {}
  with open(file_list) as file:
    files = file.readlines()
  with open(write_as, 'w') as file:
    for filename in files:
      speaker = func(filename)
      if speaker not in speaker_to_int:
        speaker_to_int[speaker] = len(speaker_to_int)
        file.write(speaker + ', ' + str(speaker_to_int[speaker]) + '\n')

def get_speaker_to_int(speaker_path):
  with open(speaker_path) as file:
    lines = file.readlines()
  speaker_to_int = {}
  for line in lines:
    speaker, number = line.strip().split(', ')
    speaker_to_int[speaker] = int(number)
  return speaker_to_int

def get_speaker_info(speaker_to_int, info_path):
  with open(info_path) as file:
    lines = file.readlines()
  speaker_info = {}
  is_vctk = '|' not in lines[0]
  is_aishell = 'aishell' in lines[0]
  for line in lines[1:]:
    speaker, info = line.split(maxsplit=1)
    if is_aishell:
      speaker = 'S' + speaker
    else:
      speaker = is_vctk * 'p' + speaker
    if speaker in speaker_to_int:
      speaker_info[speaker_to_int[speaker]] = line.strip()
  for speaker_int in speaker_to_int.values():
    if speaker_int not in speaker_info:
      speaker_info[speaker_int] = 'missing_info'
  return speaker_info

if __name__ == '__main__':
  write_speaker_to_int('vctk')
  write_speaker_to_int('librispeech')

