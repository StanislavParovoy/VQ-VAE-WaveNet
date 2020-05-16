from vqvae import VQVAE
from Encoder.encoder import *
from Decoder.WaveNet.wavenet import WaveNet
from dataset import *
from utils import *
import tensorflow as tf
import time, os, sys, datetime
from argparse import ArgumentParser

def parse_args():
  parser = ArgumentParser()
  parser.add_argument('--dataset', '-d', default='VCTK-Corpus/wav48')
  parser.add_argument('-gc', default='accent', choices=['speaker', 'accent'])
  parser.add_argument('--batch', '-b', default=8, type=int)
  parser.add_argument('--prefetch', '-p', default=64, type=int)
  parser.add_argument('--in_memory', '-im', default=False, action='store_true')
  parser.add_argument('--step', '-s', default=10000, type=int)
  parser.add_argument('--model', '-m', default='teacher', choices=['teacher', 'student'])
  parser.add_argument('--save_teacher', '-st', default='saved_teacher')
  parser.add_argument('--restore_teacher', '-rt', default=None)
  parser.add_argument('--save_student', '-ss', default='saved_student')
  parser.add_argument('--restore_student', '-rs', default=None)
  parser.add_argument('--save_interval', '-si', default=None, type=int)
  parser.add_argument('--log_interval', '-li', default=200, type=int)

  args = parser.parse_args()
  args.save_teacher = args.save_teacher.strip('/')
  args.save_student = args.save_student.strip('/')
  if args.restore_teacher is None:
    args.restore_teacher = args.save_teacher
  if args.restore_student is None:
    args.restore_student = args.save_student
  return args

def make_dataset(gc_type, **args):
  if gc_type == 'speaker':
    dataset = VCTK(**args)
  elif gc_type == 'accent':
    dataset = VCTK_Accent(**args)
  return dataset

def optimise(variables, loss, summary=True):
  learning_rate = tf.Variable(tf.constant(0, dtype=tf.float32), trainable=False, name='learning_rate')
  lr_var = learning_rate
  global_step = tf.Variable(tf.constant(0, dtype=tf.int32), trainable=False, name='global_step')
  for key, value in parameters.learning_rate_schedule.items():
    learning_rate = tf.cond(tf.less(global_step, int(key)), 
      lambda: learning_rate, lambda: tf.constant(value))

  if parameters.optimiser == 'sgd':
    optimiser = tf.train.GradientDescentOptimizer(learning_rate)
  elif parameters.optimiser == 'adam':
    optimiser = tf.train.AdamOptimizer(learning_rate)

  def clip(grad_var, summary=True):
    grad, var = grad_var
    if grad is None:
      return grad, var
    grad_name = grad.name
    grad = tf.clip_by_norm(grad, parameters.grad_clip_norm)
    if summary:
      tf.summary.histogram(grad_name, grad)
      tf.summary.histogram(var.name, var)
    return grad, var

  gradients = tf.gradients(loss, variables)
  grad_var = zip(gradients, variables)
  clipped_grad_var = list(map(clip, grad_var))
  train_op = optimiser.apply_gradients(clipped_grad_var, global_step=global_step)

  ema = None
  if parameters.ema:
    ema = tf.train.ExponentialMovingAverage(decay=0.999)
    with tf.control_dependencies([train_op]):
      train_op = ema.apply(variables)
  return train_op, global_step, learning_rate, lr_var, ema

def train(sess, saver, init, train_op, loss, global_step, args):
  # writes log for tensorboard
  save_path = args.save_teacher if args.model == 'teacher' else args.save_student
  writer = tf.summary.FileWriter(save_path, sess.graph)
  summary = tf.summary.merge_all()

  # main training loop
  gs, _ = sess.run([global_step, init])
  avg, smooth = 0, 0.3
  for step in range(1, args.step + 1):
    try:
      t = time.time()
      if gs == 1 or (gs + 1) % args.log_interval == 0:
        _, l, gs, log = sess.run([train_op, loss, global_step, summary])
        writer.add_summary(log, gs)
      else:
        _, l, gs = sess.run([train_op, loss, global_step])
      t = time.time() - t
      # avg = avg + (t - avg) / step
      avg = avg + smooth * (t - avg)
      s = int((args.step - step) * avg)
      timing = '[batch %.3fs] [ETA %s]' % (t, str(datetime.timedelta(seconds=s)))
      progress = '\r[step %d | %.2f' % (gs, step / args.step * 100) + '%]'
      info = '[loss %.4f]' % (l)
      print(' '.join((progress, info, timing)), end='            ')
    except tf.errors.OutOfRangeError:
      sess.run(init)
    if gs > 0 and args.save_interval is not None and gs % args.save_interval == 0:
      saver.save(sess, save_path + '/weights', global_step=global_step)
  if args.save_interval is None:
    saver.save(sess, save_path + '/weights', global_step=global_step)

def train_teacher(args):
  # build graph
  dataset = make_dataset(args.gc, data_path=args.dataset, batch_size=args.batch, prefetch=args.prefetch, in_memory=args.in_memory)
  # dataset.y = Upsample(trainable=True)(dataset.y)
  encoder = Encoder(scope='encoder', latent_dim=parameters.latent_dim, trainable=True)
  vqvae = VQVAE(scope='vqvae', trainable=True)
  condition = vqvae(encoder(dataset.x))
  decoder = WaveNet(args=parameters.teacher, scope='decoder', trainable=True)
  logits = decoder(x=dataset.x, condition=condition, h=dataset.y)
  # reconstruction_loss
  loss = decoder.compute_loss(logits, labels=dataset.x)
  if parameters.use_vq:
    loss += vqvae.compute_loss()

  # build optimiser on variables
  variables = tf.trainable_variables()
  train_op, global_step, learning_rate, lr_var, ema = optimise(variables, loss)

  # restore pretrained model
  os.makedirs(args.save_teacher, exist_ok=True)
  latest_teacher = tf.train.latest_checkpoint(args.restore_teacher)
  sess = tf.Session()
  saver = tf.train.Saver()
  if latest_teacher is not None:
    saver.restore(sess, latest_teacher)
  else:
    sess.run(tf.global_variables_initializer())

  gs, lr = sess.run([global_step, learning_rate])
  print('[restore %s] last global step: %d, learning rate: %.5f' % (args.model, gs, lr))
  train(sess, saver, dataset.init, train_op, loss, global_step, args)

if __name__ == '__main__':
  suppress_tf_warning()
  args = parse_args()
  if 1 or args.model == 'teacher':
    train_teacher(args)
  else:
    train_student(args)
