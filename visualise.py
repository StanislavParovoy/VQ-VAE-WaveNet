import numpy as np
import io, os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-embedding',
                    dest='embedding',
                    help='embedding space')
parser.add_argument('-speaker', 
                    dest='speaker',
                    help='speaker embedding space')
parser.add_argument('-save', 
                    dest='save',
                    help='save to folder')
args = parser.parse_args()

if not os.path.isdir(args.save):
    os.mkdir(args.save)

total = [args.embedding]
# if args.speaker:
    # total.append(args.speaker)

for file in total:
    emb = np.load(file)
    file = file.strip('.npy')
    out_v = io.open('%s/%s_vecs.tsv'%(args.save, file), 'w', encoding='utf-8')
    out_m = io.open('%s/%s_meta.tsv'%(args.save, file), 'w', encoding='utf-8')
    for i, vec in enumerate(emb):
        out_m.write(str(i+1) + "\n")
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_v.close()
    out_m.close()
print('upload to http://projector.tensorflow.org')
