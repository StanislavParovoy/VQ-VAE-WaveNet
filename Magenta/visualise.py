import sys, io
sys.path.append("..")
from masked import *
from config import Config, FastGenerationConfig
from mu_law_ops import *
from tqdm import tqdm
import numpy as np
from utils import get_speaker_to_int

if tf.__version__ == '1.14.0':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.logging.set_verbosity(tf.logging.ERROR)

config = Config()
config.build(inputs=wav, gc=None, is_training=True)
embedding = tf.get_variable(name='embedding')

gs = int(sys.argv[1])
saved_path = 'saved_vqvae_config/weights-%d'%gs
sess = tf.Session()  
saver = tf.train.Saver(config.variables)
saver.restore(sess, saved_path)

embedding = sess.run(embedding)

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for i, vec in enumerate(embedding):
  out_m.write("emb_%d\n" % i)
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()

