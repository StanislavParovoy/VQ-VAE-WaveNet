# dsp
sr = 24000
num_samples = 6656
top_db = 25
# vqvae
encoder = '64'
use_vq = True
global_embedding_dim = 64
k = 512
latent_dim = 64
beta = 0.25
# wavenet
teacher = {
	'distribution': 'gaussian',
	'min_log_scale': -8.,
	'dilations': [2 ** (i % 10) for i in range(20)],
	'kernel_size': 3,
	'dilation_filters': 256,
	'skip_filters': 256,
	'residual_filters': 256,
}
# optimiser
grad_clip_norm = 9.
ema = False
optimiser = 'adam'
learning_rate_schedule = {
  '0' : 0.0001,
  '100000' : 0.00005,
  '200000' : 0.000025,
  '300000' : 0.0000125
}