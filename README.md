
# VQ-VAE-WaveNet

This is a TensorFlow implementation of vqvae with wavenet decoder, based on https://arxiv.org/abs/1711.00937 and https://arxiv.org/abs/1901.08810.

### Dependencies:
TensorFlow r1.12 / r1.14, numpy, librosa, scipy, tqdm

### Model

#### Encoder
There are 3 encoders implemented:
- `Magenta` encoder from nsynth-magenta, wavenet alike (default)
- `2019` the one described in https://arxiv.org/abs/1901.08810
- `64` 6 layers strided conv, as mentioned in original paper

Parameters can be found in `Encoder/encoder.py` and `model_parameters.json`.

#### VQ

There are 2 ways to train the embedding:
- train $z_e$ and $e_k$ separately, as described in original paper (default)
- train them together without tf.stop_gradient

There are 2 ways to initialise the embedding:
- random normal init (default)
- identity matrix (requires k == latent_dim; need to manually change code in `model.py`; not so helpful)

This could be turned off as well, in which case an AE is trained.

Parameters can be found in `model_parameters.json`.

#### Decoder

WaveNet decoder.

Parameters can be found in `wavenet_parameters.json`.

### Training

#### Dataset

Supports VCTK (default) and LibriSpeech. 
Download data and put the unzipped folders 'VCTK-Corpus' or 'LibriSpeech' in the folder `data`.
To train from custom datasets, refer to `dataset.py` for making iterators.

example usage: 

`python3 train.py VCTK -m 0 -l 5120 -b 4 -e 1 -params model_parameters.json -log logs -interval 100 -save saved_model/weights`
- `-m` whether load data into memory or use tf io
- `-l` length of segment to use in training
- `-b` batch size
- `-e` number of epochs
- `-restore` resume from (e.g. `saved_model/weights-110640`)
- `-log` path to save logs for tensorboard
- `-interval` steps between each log written to disk
- `-save` save to (e.g. `saved_model/weights`)

### Generation

Implements fast generation; starts from zeros.

example usage:
`python3 generate.py -restore saved_model/weights-110640 -audio data/VCTK-Corpus/wav48/p225/p225_001.wav -speakers p225 p243 p292 -mode sample -save generated`
- `-restore` restore trained model
- `-audio` which audio to use as local condition
- `-speakers` which speaker(s) to use as global condition
- `-mode` method to sample from predicted quantised distribution (`sample`, `greedy`)
- `-save` where to save generated audio

### Visualisation

For now it saves the trained vq embedding space, and visualises through http://projector.tensorflow.org

example usage:
`python3 visualise.py -embedding embedding_110640.npy -save embeddings`
then upload tsv files in folder `embeddings` to the website.

### TODO
- [ ] Train a prior based on vq

### Alternative Implementation
The folder `Magenta` contains an implementation that I collaged from 'official' code. High coupling. My own implementation draws insights from there. Training and Generating are pretty similar.

### References

- https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
- https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
- https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth/wavenet
- https://github.com/ibab/tensorflow-wavenet
- https://github.com/JeremyCCHsu/vqvae-speech
