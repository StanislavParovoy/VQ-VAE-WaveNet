
# VQ-VAE-WaveNet

This is a TensorFlow implementation of vqvae with wavenet decoder, based on https://arxiv.org/abs/1711.00937 and https://arxiv.org/abs/1901.08810.

### Dependencies:
TensorFlow r1.12 - r1.15, numpy, librosa, scipy, tqdm

### Results

The folder `results` contains some reconstructed audio. Speaker conversion works well, but encoder (local condition) needs some more tuning.

### Model

#### Encoder

Implemented in `Encoder/encoder.py`, providing local condition to vocoder.

TODO: use d-vector as global condition instead of one hot id.

#### VQ

For custom encoders, initialisation of embedding may be tuned to match scale of encoder output.

VQ could be turned off by changing `use_vq=False` in `parameters.py`, in which case an AE is trained.

#### Decoder

Using WaveNet decoder. The architecture is based on ClariNet so that parallel wavenet could be trained later, which is TODO for now.

### Training

#### Dataset

Supports VCTK.
Download data and put the unzipped folders 'VCTK-Corpus' in the folder `data`.
To train from custom datasets, refer to `dataset.py` for making iterators.

example training command: 

If you have large enough ram, you can first run 
`python3 preprocess.py --dataset path_to_wav48 --save preprocessed`
then run
`python3 train.py --in_memory --dataset preprocessed -gc speaker --batch 8 --step 100000 --save_teacher saved_model`

Otherwise, run
`python3 train.py --dataset VCTK-Corpus/wav48 -gc speaker --batch 8 --step 100000 --save_teacher saved_model`

In-memory data pipeline should be more than 2x faster.

### Generation

Implements fast generation; starts from zeros.

example usage:
`python3 generate.py -restore saved_model -audio p225_001.wav -gc speaker -speakers p225 p226 p227 p228`

### References

- https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
- https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
- https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth/wavenet
- https://github.com/ibab/tensorflow-wavenet
- https://github.com/JeremyCCHsu/vqvae-speech
