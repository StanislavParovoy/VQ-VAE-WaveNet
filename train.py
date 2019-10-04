from model import VQVAE
from Encoder.encoder import *
from Decoder.decoder import *
from dataset import *
from utils import display_time
import tensorflow as tf
import time, os, sys


if tf.__version__ == '1.14.0':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.logging.set_verbosity(tf.logging.ERROR)


dataset_args = {
    'batch_size': 8,
    'in_memory': True,
    'start': None,
    'end': None,
    'shuffle': True,
    'seed': None,
    'max_len': 5120,
    'step': None, # receptive field
    'sr': 16000
}
dataset = LibriSpeech(**dataset_args)
num_batches = dataset.num_batches

wavenet_args_path = 'wavenet.json'
encoder = Encoder_Magenta()
decoder = WavenetDecoder(wavenet_args_path)
model_args = {
    'x': dataset.x,
    'speaker': dataset.y,
    'encoder': encoder,
    'decoder': decoder,
    'latent_dim': 64,
    'k': 256, 
    'beta': 0.25,
    'verbose': True
}

learning_rate_schedule = {
    0: 2e-4,
    90000: 4e-4 / 3,
    120000: 6e-5,
    150000: 4e-5,
    180000: 2e-5,
    210000: 6e-6,
    240000: 2e-6,
}

model = VQVAE(model_args)
model.build(learning_rate_schedule=learning_rate_schedule)

save_dir = 'saved_vqvae'
save_name = 'weights'

if len(sys.argv) < 2:
    restore_path = None
else:
    last_gs = int(sys.argv[1])
    restore_path = save_dir + '/' + save_name + '-%d'%last_gs

sess = tf.Session()
writer = tf.summary.FileWriter('logs_vqvae', sess.graph)
saver = tf.train.Saver()
if restore_path is not None:
    saver.restore(sess, restore_path)
else:
    sess.run(tf.global_variables_initializer())

gs = sess.run(model.global_step)
lr = sess.run(model.lr)
print('last global step: %d, learning rate: %.8f' % (gs, lr))
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

num_epochs = 4
for e in range(1, num_epochs + 1):  
    sess.run(dataset.init)
    step = 0
    while True:
        try:
            step += 1
            t = time.time()
            _, rl, vl, cl, gs, lr, summary = sess.run([model.train_op, 
                                               model.reconstruction_loss, 
                                               model.vq_loss, 
                                               model.commitment_loss,
                                               model.global_step, 
                                               model.lr,
                                               model.summary])
            writer.add_summary(summary, gs)
            t = time.time() - t
            progress = '\r[e %d step %d] %.2f' % (e, gs, step / num_batches * 100) + '%'
            loss = ' [recons %.5f] [vq %.5f] [commit %.5f] [lr %.8f]' % (rl, vl, cl, lr)
            second = (num_batches - step) * t
            print(progress + loss + display_time(t, second), end='')
        except tf.errors.OutOfRangeError:
            break
    saver.save(sess, save_dir + '/' + save_name, global_step=model.global_step)
