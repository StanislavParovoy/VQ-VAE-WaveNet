import tensorflow as tf
import parameters


class VQVAE(object):
  def __init__(self, scope, trainable=True):
    self.scope = scope

  def __call__(self, z_e):
    # local condition
    if not parameters.use_vq:
      return z_e

    # discretise z_e
    _, t, f = z_e.get_shape().as_list()
    self.codebook = tf.get_variable(
        name=self.scope + '/codebook',                         
        shape=[parameters.k, f],                          
        initializer=tf.initializers.he_uniform(seed=None))
        # initializer=tf.uniform_unit_scaling_initializer(1.4))
    # z_e: [..., d] -> [..., 1, d]
    # codebook:             [k, d]
    expanded_ze = tf.expand_dims(z_e, -2)
    distances = tf.reduce_sum((expanded_ze - self.codebook) ** 2, axis=-1)
    # q(z|x) refers to the 2d grid in the middle in figure 1
    q_z_x = tf.argmin(distances, axis=-1)
    e_k = tf.nn.embedding_lookup(self.codebook, q_z_x)

    tf.summary.histogram('distances', distances)
    tf.summary.histogram('q(z|x)', q_z_x)
    tf.summary.histogram('e_k', e_k)

    # this should pass gradient from z_q to z_e
    # evaluates to: z_q = tf.gather(params=codebook, indices=q_z_x)
    z_q = z_e + tf.stop_gradient(e_k - z_e)

    self.z_e = z_e
    self.e_k = e_k
    return z_q

  def compute_loss(self):
    vq_loss = tf.reduce_mean((tf.stop_gradient(self.z_e) - self.e_k) ** 2)
    tf.summary.scalar('vq_loss', vq_loss)
    commitment_loss = tf.reduce_mean((self.z_e - tf.stop_gradient(self.e_k)) ** 2)
    return vq_loss + parameters.beta * commitment_loss

