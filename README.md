<!--
SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>

SPDX-FileContributor: Alexandre Bittar abittar@idiap.ch>

SPDX-License-Identifier: BSD-3-Clause
--->

# SPARSE: Spiking Architectures towards Realistic Speech Encoding

## Description

This software defines surrogate gradient spiking neural networks (SNNs) that result
in physiologically inspired speech encoders. The PyTorch modules can be easily
integrated into deep learning frameworks to be trained and analysed on automatic
speech recognition (ASR) tasks. We also provide scripts to run training and analysis
inside the [SpeechBrain](https://github.com/speechbrain/speechbrain) framework.

## Installation
```
git clone git@gitlab.idiap.ch:abittar/sparse.git
cd sparse
pip install -r requirements.txt
pip install -e .
```

## General utilisation

The PyTorch modules defined in the `sparse` folder can directly be used in a
Python script. For instance, you can build an SNN using

```
import torch
import torch.nn as nn
from sparse.snn import MultilayeredSpiking

# Generate random input
batch_size, num_steps, input_size = 8, 500, 144
x = (torch.rand(batch_size, num_steps, input_size) > 0.9).float()

# Instantiate module
net = MultilayeredSpiking(
    input_size=input_size,
    layer_sizes=[144, 144, 10],
    sfa_portions=[0.5, 0.5, 0.5],
    rnn_portions=[0.5, 0.5, 0.5],
    ff_portions=[1.0, 1.0, 1.0],
    dropout=0.1,
    dt=1.0,
)

# Pass input through module
y = net(x)
```

Similarly, you can build an SNN-based encoder for ASR as
```
import torch
import torch.nn as nn
from sparse.encoder_asr import ASREncoder

# Define random input
batch_size, num_frames, num_mels = 8, 1000, 80
x = torch.rand(batch_size, num_frames, num_mels)

# Instantiate module
model = ASREncoder(
    num_mel_bins=num_mels,
    auditory_channels=[16],
    auditory_kernel_size=(7, 7),
    auditory_frequency_stride=[1],
    cortex_layer_sizes=[512]*3,
    cortex_sfa_portions=[0.5]*3,
    cortex_rnn_portions=[0.5]*3,
    cortex_feedforward_portions=[1.0]*3,
    phoneme_num_layers=2,
    phoneme_features=512,
    phoneme_ctc_rate_hz=25,
    dropout=0.15,
    dt=1.0,
)

# Pass input through module
y = model(x)
```

## Utilisation inside SpeechBrain

We used the [SpeechBrain](https://github.com/speechbrain/speechbrain) framework
to run our experiments on ASR tasks. Here we provide the instructions necessary
to integrate our models inside SpeechBrain and run training and analysis
scripts on the TIMIT dataset with CTC decoding. The same could be done on other
ASR recipes.

First you need to git clone and install SpeechBrain as,
```
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install --editable .
```

You can then place the `run_training.py` and `configs/timit_train.yaml` from the
`sparse` package inside the SpeechBrain CTC-based ASR recipe for the TIMIT dataset at
`recipes/TIMIT/ASR/CTC`. Note that your path to the TIMIT dataset must be specified
in the config file by replacing the paceholder with `!ref /your/path/to/timit/`.
```
cp /path/to/sparse/run_training.py /path/to/speechbrain/recipes/TIMIT/ASR/CTC/run_training.py
cp /path/to/sparse/configs/timit_train.yaml /path/to/speechbrain/recipes/TIMIT/ASR/CTC/hparams/timit_train.yaml
```
You should then be able to run training from SpeechBrain as
```
cd /path/to/speechbrain/recipes/TIMIT/ASR/CTC
python run_training.py hparams/timit_train.yaml
```
Once you have a trained model, you can perform the analysis by similarly placing the `run_analysis.py` and `configs/timit_analysis.yaml` from the
`sparse` package into the SpeechBrain recipe: 
```
cp /path/to/sparse/run_analysis.py /path/to/speechbrain/recipes/TIMIT/ASR/CTC/run_analysis.py
cp /path/to/sparse/configs/timit_analysis.yaml /path/to/speechbrain/recipes/TIMIT/ASR/CTC/hparams/timit_analysis.yaml
```
and then run it from SpeechBrain as,
```
cd /path/to/speechbrain/recipes/TIMIT/ASR/CTC
python run_analysis.py hparams/timit_analysis.yaml
```
Note that you have to use the same output folder and encoder parameters in your analysis config as in your training config.

## Description

### SNN
We use the linear adaptive leaky integrate-and-fire model (AdLIF) for our spiking neurons. A proportion of neurons can be reduced to standard LIF without adaptation. For instance, a layer with `sfa_portion=0.5` has half of its neurons LIF and the other half AdLIF. 

The feedforward and recurrent connectivity of a layer can also be reduced. For instance, using `ff_portion=1.0` and `rnn_portion=0.5` yields a layer with fully-connected feedforward weight matrix but with a recurrent matrix where a randomly chosen half of its entries are masked to zero.

### ASR encoder
The ASR encoder receives acoustic features and outputs phonemic features that must additionally be projected to phoneme or subword classes. It consists of a 2D-convolutional layer with LIF neurons which simulates auditory nerve fibers, followed by a multi-layered recurrent SNN using AdLIF neurons. The spike trains of the last layer are average-pooled
over the time dimension and projected to phoneme features. The non-spiking parts of the encoder are kept simple so that the main processing is done by the spiking neurons.

### KWS encoder
For keyword spotting, the average-pooling of the ASR encoder is replaced by a readout layer, which gets rid of the time dimension. The readout layer is defined as a non-recurrent layer of non-spiking LIF neurons. Their membrane potential is then summed using the softmax function at each time step so that the output represents a measure of each neuron's activity over the complete sequence compared to the others. 

### Analysis
The analysis can produce the following plots for the model:
- Distribution of neuron parameter values
- Distribution of weight matrix values

For a chosen utterance, it also produces the following plots:
- Spike train raster plot 
- Distribution of single neuron firing rates
- Population signal as the normalised sum of spike trains in a layer, filtered in different frequency bands.
- Phase amplitude coupling (PAC) between two specified populations (can be the same) using the modulation index and mean vector length metrics.

When specified, it also tests the significance of PAC between all possible populations and frequency bands on all utterances detailed in an output text file.

