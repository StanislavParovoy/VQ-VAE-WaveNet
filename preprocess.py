import numpy as np
import librosa, os, soundfile, datetime, time
from tqdm import tqdm
from scipy.io import wavfile
import parameters
from argparse import ArgumentParser

def parse_args():
  parser = ArgumentParser()
  parser.add_argument('--dataset', default='../VCTK-Corpus/wav48', type=str)
  parser.add_argument('--filetype', default='wav', type=str)
  parser.add_argument('--save', default='vctk_preprocessed', type=str)
  args = parser.parse_args()
  args.save = args.save.strip('/')
  return args

def get_file_and_speaker(data_path, depth, filetype):
  stack = [data_path.strip('/')]
  # level-by-level bfs
  for i in range(depth):
    new_stack = []
    for parent in stack:
      for child in os.listdir(parent):
        new_stack.append(parent + '/' + child)
    stack = new_stack
  total = []
  for filename in stack:
    if filename.endswith(filetype):
      speaker = filename.split('/')[-depth]
      total.append((filename, speaker))
  print('num files total:', len(total))
  return total

def preprocess(wav):
  wav, index = librosa.effects.trim(wav, top_db=parameters.top_db)
  return wav

def file_to_npy(file):
  wav, _ = librosa.load(file, sr=parameters.sr)
  wav = preprocess(wav)
  return wav

def preprocess_all(args):
  file_and_speaker = get_file_and_speaker(args.dataset, 2, args.filetype)
  for file, speaker in tqdm(file_and_speaker):
    name = file[file.rfind('/'): file.find('.' + args.filetype)]
    save_name = args.save + '/' + speaker + '/' + name + '.npy'
    if os.path.isfile(save_name):
      continue
    os.makedirs(args.save + '/' + speaker, exist_ok=True)
    s = file_to_npy(file, )
    np.save(save_name, s)

if __name__ == '__main__':
  args = parse_args()
  preprocess_all(args)

