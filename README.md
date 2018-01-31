# A TensorFlow implementation of VQVAE paper

This is a TensorFlow implementation of the [Vector Quantised-Variational AutoEncoder](https://papers.nips.cc/paper/7210-neural-discrete-representation-learning.pdf) for voice conversion.

## Features
- VQVAE for voice conversion
- WaveNet which allows local conditioning

## Requirements
Code is tested on TensorFlow version 1.4 for Python 3.6.

In addition, [librosa](https://github.com/librosa/librosa) must be installed for reading and writing audio.

## Simple test
<p>In this test, we generate 3-type wavs and each type has different style each other.</p>

<p>See below generated examples for train.</p>
<img src="results/type_1_1.png" />
<img src="results/type_1_2.png" />
<img src="results/type_2_1.png" />
<img src="results/type_2_2.png" />
<img src="results/type_3_1.png" />
<img src="results/type_3_2.png" />

### Results
<img src="results/test_source1.png" />
<img src="results/test_source1_to_type2.png" />
<img src="results/test_source1_to_type3.png" />
<img src="results/test_source2.png" />
<img src="results/test_source2_to_type1.png" />
<img src="results/test_source2_to_type3.png" />

You can find more details in <a href="./notebooks/simple vq-vae test.ipynb">simple vq-vae test</a> and the generated files are on <a href="./results">results</a> folder.

## Voice conversion test
This will be updated soon.

## References
- [VQVAE](https://github.com/hiwonjoon/tf-vqvae), a VQVAE for image generation
- [wavenet](https://github.com/ibab/tensorflow-wavenet), a wavenet


