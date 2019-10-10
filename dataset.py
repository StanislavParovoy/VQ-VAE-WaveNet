import tensorflow as tf
import numpy as np
import librosa, json
from tqdm import tqdm
from scipy.io import wavfile
from utils import get_speaker_to_int


class Dataset():
    def __init__(self):
        self.x = None
        self.y = None
        self.init = None
        self.all_files = None
        self.speaker_to_int = None

    def read_files(self, filename, start=None, end=None, shuffle=True, seed=None):
        with open(filename) as file:
            files = file.readlines()
        total = len(files)
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(files)
        if end is not None:
            files = files[: end]
        if start is not None:
            files = files[start: ]
        if start is None:
            start = 0
        if end is None:
            end = total
        print('data: [%d -> %d] [%.2f' % (start, end, 100 * end / total) + '%' + ' so far]')
        return [f.strip() for f in files]

    def trim_silence(self, audio, threshold=0.01, frame_length=2048):
        '''Removes silence at the beginning and end of a sample.'''
        if audio.size < frame_length:
            frame_length = audio.size
        energy = librosa.feature.rms(audio, frame_length=frame_length)
        frames = np.nonzero(energy > threshold)
        indices = librosa.core.frames_to_samples(frames)[1]
        # Note: indices can be an empty array, if the whole audio was silence.
        return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    def generator(self, data, labels):
        def gen():
            for x, y in zip(data, labels):
                yield x, y
        return gen

    @staticmethod
    def read(filename, speaker, max_len, sr, abs_name=''):
        contents = tf.io.read_file(abs_name + filename)
        wav = tf.contrib.ffmpeg.decode_audio(contents, 'wav', sr, 1)
        wav = tf.random_crop(wav, [max_len, 1])
        return wav, speaker

    def _load(self, max_len, step, split_func, abs_name=''):
        data = np.zeros([len(self.all_files), max_len, 1], dtype=np.float32)
        speakers = np.zeros([len(self.all_files)], dtype=np.float32)

        for n, file in tqdm(enumerate(self.all_files)):
            sr, wav = wavfile.read(abs_name + file)
            wav = np.asarray(wav, dtype=np.float32)
            if len(np.shape(wav)) > 1:
                wav = (wav[:, 0] + wav[:, 1]) / 2
            wav = (wav + 0.5) / 32767.5
            wav = self.trim_silence(wav)
            if len(wav) < max_len:
                continue
            wav = np.expand_dims(wav, -1)

            speaker = split_func(file)
            speaker_id = self.speaker_to_int[speaker]

            i = np.random.randint(0, len(wav) - max_len + 1)
            data[n] = wav[i: i + max_len]
            speakers[n] = speaker_id

        speakers = np.reshape(speakers, [-1, 1])
        print('data total:', len(data))
        return data, len(data), speakers

    def _get_speakers(self, max_len, split_func):
        speakers = []
        for file in self.all_files:
            speaker = split_func(file)
            speaker_id = self.speaker_to_int[speaker]
            speakers.append(speaker_id)
        return np.reshape(speakers, [-1, 1])


class LibriSpeech(Dataset):
    def __init__(self, batch_size=8, in_memory=True, 
        relative_path='', start=None, end=None, 
        shuffle=True, seed=0, max_len=5120, step=None, sr=16000):
        super(LibriSpeech, self).__init__()

        filename='librispeech_train_clean_100.txt'
        speaker_file = 'librispeech_speakers.txt'
        data_dir = ''
        if relative_path != '':
            filename = relative_path + filename
            speaker_file = relative_path + speaker_file
            data_dir = relative_path + data_dir

        self.speaker_to_int = get_speaker_to_int(speaker_file)
        self.all_files = self.read_files(filename, start, end, shuffle, seed)
        split_func = lambda s: s.split('/')[-1].split('-', 1)[0]

        if in_memory:
            self.data, self.total, self.speakers = self._load(max_len, step, split_func, data_dir)
            gen = self.generator(self.data, self.speakers)
            dataset = tf.data.Dataset.from_generator(gen, 
                (tf.float32, tf.int32), ([max_len, 1], [1]))
            dataset = dataset.shuffle(self.total)
        else:
            self.speakers = self._get_speakers(max_len, split_func)
            self.total = len(self.all_files)
            dataset = tf.data.Dataset.from_tensor_slices((self.all_files, self.speakers))
            dataset = dataset.shuffle(self.total)
            dataset = dataset.map(map_func=lambda x, y: self.read(x, y, max_len, sr, data_dir), 
                                  num_parallel_calls=4)

        dataset = dataset.batch(batch_size, drop_remainder=False)
        iterator = dataset.make_initializable_iterator()
        self.init = iterator.initializer

        self.x, y = iterator.get_next()
        self.y = tf.one_hot(y, depth=40, dtype=tf.float32)
        self.num_batches = np.ceil(self.total / batch_size)


class VCTK(Dataset):
    def __init__(self, batch_size=8, in_memory=True, relative_path='', 
        start=None, end=None, shuffle=True, seed=0, max_len=5120, step=None, sr=16000):
        super(VCTK, self).__init__()

        filename = 'vctk_train.txt'
        speaker_file = 'vctk_speakers.txt'
        data_dir = 'VCTK-Corpus/wav48/'
        if relative_path != '':
            filename = relative_path + filename
            speaker_file = relative_path + speaker_file
            data_dir = relative_path + data_dir
        self.speaker_to_int = get_speaker_to_int(speaker_file)
        self.all_files = self.read_files(filename, start, end, shuffle, seed)
        split_func = lambda s: s.split('/')[0]

        if in_memory:
            self.data, self.total, self.speakers = self._load(max_len, step, split_func, data_dir)
            gen = self.generator(self.data, self.speakers)
            dataset = tf.data.Dataset.from_generator(gen, 
                (tf.float32, tf.int32), ([max_len, 1], [1]))
            dataset = dataset.shuffle(self.total)
        else:
            self.speakers = self._get_speakers(max_len, split_func)
            self.total = len(self.all_files)
            dataset = tf.data.Dataset.from_tensor_slices((self.all_files, self.speakers))
            dataset = dataset.shuffle(self.total)
            dataset = dataset.map(
                map_func=lambda x, y: self.read(x, y, max_len, sr, data_dir), 
                num_parallel_calls=4)

        dataset = dataset.batch(batch_size, drop_remainder=False)
        iterator = dataset.make_initializable_iterator()
        self.init = iterator.initializer

        self.x, y = iterator.get_next()
        self.y = tf.one_hot(y, depth=109, dtype=tf.float32)
        self.num_batches = np.ceil(self.total / batch_size)


class NSynth(Dataset):
    def __init__(self, batch_size=8, in_memory=True, **kwargs):
        super(NSynth, self).__init__()
        if in_memory:
            self.data, self.instruments, self.total = self._np_load(**kwargs)

        # dataset = tf.data.Dataset.from_tensor_slices((self.data, self.instruments))
        gen = self.generator(self.data, self.instruments)
        dataset = tf.data.Dataset.from_generator(gen, 
            (tf.float32, tf.int32), ([None, 1], [None]))
        dataset = dataset.shuffle(len(self.data)).batch(batch_size, drop_remainder=False)
        iterator = dataset.make_initializable_iterator()
        self.init = iterator.initializer

        self.x, y = iterator.get_next()
        self.y = tf.one_hot(y, depth=11, dtype=tf.float32)
        self.num_batches = np.ceil(self.total / batch_size)


    def _np_load(self, data_dir):
        with open(data_dir + '/examples.json') as file:
            files = json.load(file)
        sr = 16000
        c = 32767.5
        data = []
        labels = []
        for i, k in tqdm(enumerate(files.keys())):
            sr, wav = wavfile.read('nsynth-valid/audio/'+k+'.wav')
            wav = (wav[:sr * 4] + 0.5) / c
            data.append(wav)
            instrument = files[k]['instrument_family']
            labels.append(np.repeat(instrument, len(wav)))
        print('total:', len(data))
        return np.expand_dims(data, -1), np.asarray(labels), len(data)






