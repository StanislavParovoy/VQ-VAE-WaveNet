import numpy as np
import io, os
from argparse import ArgumentParser
from utils import get_speaker_to_int, get_speaker_info

parser = ArgumentParser()
parser.add_argument('-embedding',
                    dest='embedding',
                    help='embedding space')
parser.add_argument('-speaker', 
                    dest='speaker',
                    help='speaker embedding space')
parser.add_argument('-dataset', default='VCTK',
                    dest='dataset',
                    help='VCTK or LibriSpeech')
parser.add_argument('-save', 
                    dest='save',
                    help='save to folder')
args = parser.parse_args()

if not os.path.isdir(args.save):
    os.mkdir(args.save)

total, meta = [], []
if args.embedding:
    total.append(args.embedding)
    meta.append(lambda i: str(i+1) + '\n')
if args.speaker:
    total.append(args.speaker)
    if args.dataset == 'VCTK':
        speaker_path = 'data/vctk_speakers.txt'
        info_path = 'data/vctk_speaker_info.txt'
    elif args.dataset == 'LibriSpeech':
        speaker_path = 'data/librispeech_speakers.txt'
        info_path = 'data/librispeech_speaker_info.txt'
    speaker_to_int = get_speaker_to_int(speaker_path)
    speaker_info = get_speaker_info(speaker_to_int, info_path)
    meta.append(lambda i: speaker_info[i] + '\n')

for file, id_f in zip(total, meta):
    emb = np.load(file)
    file = file.split('/')[-1].strip('.npy')
    out_v = io.open('%s/%s_vecs.tsv'%(args.save, file), 'w', encoding='utf-8')
    out_m = io.open('%s/%s_meta.tsv'%(args.save, file), 'w', encoding='utf-8')
    for i, vec in enumerate(emb):
        out_m.write(id_f(i))
        out_v.write('\t'.join([str(x) for x in vec]) + '\n')
    out_v.close()
    out_m.close()
print('upload to http://projector.tensorflow.org')
