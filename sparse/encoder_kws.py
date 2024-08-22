#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
"""
This is where an SNN-based encoder is defined for keyword
spotting (KWS) tasks.
"""
import torch
import torch.nn as nn

from sparse.encoder_utils import MergeFeatsAndChannels
from sparse.encoder_utils import Permute
from sparse.encoder_utils import print_num_params
from sparse.snn import MultilayeredSpiking
from sparse.snn import ReadoutLayer
from sparse.snn import SpikingNeurons


class KWSEncoder(nn.Module):
    """
    The input represents Mel filterbank acoustic features with shape
    (batch, time, num_mel_bins).

    The features are converted to spikes by a 2d convolutional layer
    followed by LIF neurons, simulating auditory nerve fibers.

    The resulting spike trains enter a multi-layered recurrent SNN of
    AdLIF neurons that represents an aggregate of higher order cortical
    processing. The feedforward and recurrent connectivity can be
    controlled as well as the proportion of AdLIF vs LIF neurons.

    In order to output a single probability per keyword class, the spike
    trains from the last cortical layer pass through a LIF readout layer
    which reduces the temoral dimension.

    Arguments:
    ----------
    num_mel_bins : int
        Number of input features corresponding to Mel frequency bins.
    auditory_channels : int or list of int
        Number of convolution channels.
    auditory_kernel_size : int tuple
        Kernel size of 2d convolution with shape (time, feats).
    auditory_frequency_stride : int or list of int
        Stride of convolution on feature dimension.
    auditory_sfa_portion : float
        Portion of AdLIF auditory nerve fibers, others are reduced to LIF.
    auditory_rnn_portion : float
        Portion of nonzero recurrent connections between auditory nerve fibers.
    auditory_use_lateral_rnn : bool
        If True, only nearby neurons are recurrently connected within a radius,
        where radius=rnn_portion*layer_size. If False, the nonzero elements of
        the recurrent matrix are randomly chosen based on rnn_portions.
    cortex_layer_sizes : list of int
        Number of cortical neurons in different hidden layers.
    cortex_feedforward_portions : list of floats
        Proportion of nonzero feedforward connections between cortical neurons
        in different layers.
    cortex_sfa_portions : list of floats
        Proportion of AdLIF cortical neurons in different layer, the others are
        reduced to LIF.
    cortex_rnn_portions : list of floats
        Proportion of nonzero recurrent connections between cortical neurons
        in different layers.
    cortex_use_lateral_rnn : bool
        If True, only nearby neurons are recurrently connected within a radius,
        where radius=rnn_portion*layer_size. If False, the nonzero elements of
        the recurrent matrix are randomly chosen based on rnn_portions.
    readout_features : int
        Number of readout classes or features.
    activation : torch.nn module
        Activation function used in CNN and FC layers.
    dropout : float
        Dropout probability between 0 and 1.
    adlif_tauu_lim, adlif_tauw_lim, adlif_a_lim, adlif_b_lim : float lists
        Range of allowed values for the AdLIF neuron parameters.
    adlif_threshold : float
        Spike threshold value, should be 1.0 but is kept as a variable.
    adlif_use_normal_init : bool
        Whether to use normal or uniform initialisation for AdLIF parameters.
    dt : float
        Timestep of the simulation in ms (tauu and tauw are also in ms).
    """

    def __init__(
        self,
        num_mel_bins=80,
        auditory_channels=[16],
        auditory_kernel_size=(7, 7),
        auditory_frequency_stride=[1],
        auditory_sfa_portion=0.0,
        auditory_rnn_portion=0.0,
        auditory_use_lateral_rnn=True,
        cortex_layer_sizes=[512, 512, 512],
        cortex_feedforward_portions=[1.0, 1.0, 1.0],
        cortex_rnn_portions=[0.5, 0.5, 0.5],
        cortex_sfa_portions=[0.5, 0.5, 0.5],
        cortex_use_lateral_rnn=False,
        readout_features=512,
        activation=nn.LeakyReLU,
        dropout=0.15,
        adlif_tauu_lim=[3.0, 25.0],
        adlif_tauw_lim=[30.0, 350.0],
        adlif_a_lim=[-0.5, 5.0],
        adlif_b_lim=[0.0, 2.0],
        adlif_use_normal_init=False,
        adlif_threshold=1.0,
        dt=1.0,
    ):
        super().__init__()

        if type(auditory_channels) is int:
            auditory_channels = [auditory_channels]
        if type(auditory_frequency_stride) is int:
            auditory_frequency_stride = [auditory_frequency_stride]
        if len(auditory_channels) not in [1, 2]:
            raise NotImplementedError("Number of CNN layers must be 1 or 2.")
        self.cortex_num_layers = len(cortex_layer_sizes)
        self.cortex_layer_sizes = cortex_layer_sizes
        self.unused_param_count = 0
        self.metrics = {}

        # Auditory CNN
        num_feats_after_cnn = (
            num_mel_bins - auditory_kernel_size[1]
        ) // auditory_frequency_stride[0] + 1
        self.conv_auditory = [
            nn.Conv2d(
                in_channels=1,
                out_channels=auditory_channels[0],
                kernel_size=auditory_kernel_size,
                stride=(1, auditory_frequency_stride[0]),
                padding=(auditory_kernel_size[0], 0),
            ),  # (batch, chan, time, feats)
            Permute(shape=(0, 2, 3, 1)),  # (batch, time, feats, chan)
            nn.LayerNorm(normalized_shape=[num_feats_after_cnn, auditory_channels[0]]),
            activation(),
        ]
        if len(auditory_channels) == 2:
            num_feats_after_cnn = (
                num_feats_after_cnn - auditory_kernel_size[1]
            ) // auditory_frequency_stride[1] + 1
            self.conv_auditory += [
                Permute(shape=(0, 3, 1, 2)),  # (batch, chan, time, feats)
                nn.Conv2d(
                    in_channels=auditory_channels[0],
                    out_channels=auditory_channels[1],
                    kernel_size=auditory_kernel_size,
                    stride=(1, auditory_frequency_stride[1]),
                    padding=(auditory_kernel_size[0], 0),
                ),  # (batch, chan, time, feats)
                Permute(shape=(0, 2, 3, 1)),  # (batch, time, feats, chan)
                nn.LayerNorm(
                    normalized_shape=[num_feats_after_cnn, auditory_channels[1]]
                ),
                activation(),
            ]
        self.conv_auditory += [
            Permute(shape=(0, 2, 3, 1)),  # (batch, feats, chan, time)
            nn.Dropout2d(p=dropout),
            Permute(shape=(0, 3, 1, 2)),  # (batch, time, feats, chan)
            MergeFeatsAndChannels(),  # (batch, time, feats*chan)
        ]
        self.conv_auditory = nn.Sequential(*self.conv_auditory)

        # Auditory SNN
        self.num_auditory_nerve_fibers = auditory_channels[-1] * num_feats_after_cnn
        self.snn_auditory = SpikingNeurons(
            input_size=self.num_auditory_nerve_fibers,
            sfa_portion=auditory_sfa_portion,
            rnn_portion=auditory_rnn_portion,
            use_lateral_rnn=auditory_use_lateral_rnn,
            dropout=dropout,
            tauu_lim=adlif_tauu_lim,
            tauw_lim=adlif_tauw_lim,
            a_lim=adlif_a_lim,
            b_lim=adlif_b_lim,
            use_normal_init=adlif_use_normal_init,
            threshold=adlif_threshold,
            dt=dt,
        )
        self.unused_param_count += self.snn_auditory.unused_param_count

        # Cortex SNN
        self.snn_cortex = MultilayeredSpiking(
            input_size=self.num_auditory_nerve_fibers,
            layer_sizes=cortex_layer_sizes,
            sfa_portions=cortex_sfa_portions,
            rnn_portions=cortex_rnn_portions,
            ff_portions=cortex_feedforward_portions,
            use_lateral_rnn=cortex_use_lateral_rnn,
            dropout=dropout,
            use_bias=True,
            normalization="batchnorm",
            tauu_lim=adlif_tauu_lim,
            tauw_lim=adlif_tauw_lim,
            a_lim=adlif_a_lim,
            b_lim=adlif_b_lim,
            use_normal_init=adlif_use_normal_init,
            threshold=adlif_threshold,
            dt=dt,
            return_all_spikes=True,
        )
        self.unused_param_count += self.snn_cortex.unused_param_count

        # Readout: spike trains to vector w/o time dimension
        self.readout = ReadoutLayer(
            input_size=cortex_layer_sizes[-1],
            hidden_size=readout_features,
            dropout=0.0,
            normalization="batchnorm",
            use_bias=True,
        )

        # Print trainable parameters in different modules
        print_num_params(self.conv_auditory, "CNN auditory")
        print_num_params(self.snn_auditory, "SNN auditory")
        print_num_params(self.snn_cortex, "SNN cortex")
        print_num_params(self.readout, "Readout")
        print_num_params(self, "Encoder (total)")

    def forward(self, feats):

        # Initialize output spikes for analysis
        all_spikes = []

        # Encoder acoustic features into spikes
        x = self.conv_auditory(feats.unsqueeze(dim=1))
        x = self.snn_auditory(x)
        all_spikes.append(x)

        # Higher level processing
        x, cortex_spikes = self.snn_cortex(x)
        all_spikes = all_spikes + cortex_spikes

        # Decode spikes into vector w/o time dimension
        x = self.readout(x)

        return x, all_spikes


if __name__ == "__main__":

    # Define random input
    batch_size, num_frames, num_mels = 8, 1000, 80
    x = torch.rand(batch_size, num_frames, num_mels)

    # Define model
    model = KWSEncoder(
        num_mel_bins=80,
        auditory_channels=[16],
        auditory_kernel_size=(7, 7),
        auditory_frequency_stride=[1],
        cortex_layer_sizes=[512] * 3,
        cortex_sfa_portions=[0.5] * 3,
        cortex_rnn_portions=[0.5] * 3,
        cortex_feedforward_portions=[1.0] * 3,
        readout_features=512,
        dropout=0.15,
        dt=1.0,
    )

    # Pass input through model
    y, all_spikes = model(x)
