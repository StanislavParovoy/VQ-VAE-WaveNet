# VQ-VAE-WaveNet

Dependencies:

TensorFlow r1.12 or r1.14
librosa
scipy
tqdm

This is a TensorFlow implementation of vqvae, based on https://arxiv.org/abs/1711.00937 and https://arxiv.org/abs/1901.08810.

There are 3 encoders implemented:
6 layers strided conv, as mentioned in original paper;
the one used in nsynth-magenta; (default)
the one described in https://arxiv.org/abs/1901.08810

The folder Magenta contains an implementation that I collaged from 'official' code. My own implementation is modified from there as well.

Dataset:
Supports Librispeech and VCTK. Put the folders 'LibriSpeech' or 'VCTK-Corpus' in the folder data. Default is VCTK.

TODO:
Tune.
Fix generation to multiple speakers at a time.
Add visualisation of learnt vq.
Train a prior based on vq.

Reference repositories:

https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth/wavenet
https://github.com/ibab/tensorflow-wavenet
https://github.com/JeremyCCHsu/vqvae-speech