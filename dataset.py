import tensorflow as tf
import numpy as np
import librosa, json, os
from tqdm import tqdm
from scipy.io import wavfile
from utils import get_speaker_to_int
from preprocess import file_to_npy, get_file_and_speaker
import parameters

def trim_silence(audio, threshold=0.01, frame_length=2048):
  '''Removes silence at the beginning and end of a sample.'''
  if audio.size < frame_length:
    frame_length = audio.size
  energy = librosa.feature.rms(audio, frame_length=frame_length)
  frames = np.nonzero(energy > threshold)
  indices = librosa.core.frames_to_samples(frames)[1]
  # Note: indices can be an empty array, if the whole audio was silence.
  return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

class Dataset():
  def __init__(self, data_path, batch_size, prefetch, in_memory):
    self.data_path = data_path
    self.batch_size = batch_size
    self.prefetch = prefetch
    self.in_memory = in_memory
    self.make_iterator()

  def _get_file_and_speaker(self):
    raise NotImplementedError

  def generator(self):
    raise NotImplementedError

  @property
  def gc_dim(self):
    raise NotImplementedError

  def make_iterator(self):
    gen = self.generator()
    dataset = tf.data.Dataset.from_generator(gen, 
      (tf.float32, tf.int32), ([parameters.num_samples, 1], [1]))

    dataset = dataset.batch(self.batch_size, drop_remainder=True)
    if self.prefetch is not None and self.prefetch > 0:
      dataset = dataset.prefetch(self.prefetch)
    iterator = dataset.make_initializable_iterator()
    self.init = iterator.initializer
    self.x, y = iterator.get_next()
    self.y = tf.one_hot(y, depth=self.gc_dim, dtype=tf.float32)


class VCTK(Dataset):
  def __init__(self, **kwargs):
    # self.speaker_file = 'data/vctk_info/vctk_speakers.txt'
    self.speaker_file = 'vctk_speakers.txt'
    self.speaker_to_int = get_speaker_to_int(self.speaker_file)
    super(VCTK, self).__init__(**kwargs)

  @property
  def gc_dim(self):
    print('num gc:', len(self.speaker_to_int))
    return len(self.speaker_to_int)

  def generator(self):
    file_and_speaker = self._get_file_and_speaker()
    indices = list(range(len(file_and_speaker)))
    if self.in_memory:
      cache = {}
      for filename, _ in tqdm(file_and_speaker):
        npy_file = filename.replace('.wav', '.npy')
        cache[filename] = np.load(npy_file)
    def gen():
      while True:
        i = np.random.choice(indices)
        filename, speaker = file_and_speaker[i]
        speaker_id = np.reshape(self.speaker_to_int[speaker], [1])
        if self.in_memory:
          wav = cache[filename]
        else:
          wav = file_to_npy(filename)
        if len(wav) < parameters.num_samples:
          continue
        start = np.random.randint(low=0, high=len(wav) - parameters.num_samples)
        wav = wav[start: start + parameters.num_samples]
        wav = np.expand_dims(wav, -1)
        yield wav, speaker_id
    return gen

  def _get_file_and_speaker(self):
    return get_file_and_speaker(self.data_path, 2, 'npy' if self.in_memory else 'wav')


class VCTK_AccentID:
  def __init__(self, path):
    s2i = dict()
    a2i = dict()
    i2a = []
    with open(path) as file:
      head = file.readline().split()
      s_i = head.index('ID')
      a_i = head.index('ACCENTS')
      for line in file.readlines():
        line = line.split()
        s = line[s_i]
        a = line[a_i]
        if a not in a2i:
          a2i[a] = len(a2i)
          i2a.append(a)
        s2i[s] = a2i[a]
    s2i['280'] = 0 # missing info for this person
    self.s2i = s2i # speaker (225) to sparse index (0)
    self.a2i = a2i # accent (English) to sparse index (0)
    self.i2a = i2a # sparse index (0) to accent (English)
    self.total_a = len(a2i)

  def name_to_accent_id(self, name):
    assert name in self.s2i, 'speaker %s \'s accent not found in training dataset' % name
    return self.s2i[name]


class VCTK_Accent(VCTK):
  def __init__(self, **kwargs):
    accent_path = kwargs['data_path'].replace('/wav48', '') + '/speaker-info.txt'
    self.accent_id = VCTK_AccentID(accent_path)
    super(VCTK, self).__init__(**kwargs)

  @property
  def gc_dim(self):
    print('num gc:', self.accent_id.total_a)
    return self.accent_id.total_a

  def generator(self):
    file_and_speaker = self._get_file_and_speaker()
    indices = list(range(len(file_and_speaker)))
    if self.in_memory:
      cache = {}
      for filename, _ in tqdm(file_and_speaker):
        npy_file = filename.replace('.wav', '.npy')
        cache[filename] = np.load(npy_file)
    def gen():
      while True:
        i = np.random.choice(indices)
        filename, speaker = file_and_speaker[i]
        accent_id = self.accent_id.name_to_accent_id(speaker.strip('p'))
        accent_id = np.reshape(accent_id, [1])
        if self.in_memory:
          wav = cache[filename]
        else:
          wav = file_to_npy(filename)
        if len(wav) < parameters.num_samples:
          continue
        start = np.random.randint(low=0, high=len(wav) - parameters.num_samples)
        wav = wav[start: start + parameters.num_samples]
        wav = np.expand_dims(wav, -1)
        yield wav, accent_id
    return gen

