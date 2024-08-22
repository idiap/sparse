#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
"""
This is where spiking neural network (SNN) modules are defined.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse.snn_utils import SpikeFunctionBoxcar
from sparse.snn_utils import ZeroTimes
from sparse.snn_utils import init_connection_mask
from sparse.snn_utils import init_sfa_mask


class MultilayeredSpiking(nn.Module):
    """
    Multi-layered Spiking Neural Network (SNN) where input and output
    tensors have shape (batch, time, feats).

    Arguments
    ---------
    input_size : tuple
        Number of input features.
    layer_sizes : int list
        Number of neurons in all hidden layers.
    sfa_portions : list of floats
        Portions of neurons in each layer with spike frequency adaptation.
    rnn_portions : list of floats
        Portions of nonzero recurrent connections in each layer.
    use_lateral_rnn : bool or list of bool
        If True, only nearby neurons are recurrently connected within a radius,
        where radius=rnn_portion*layer_size. If False, the nonzero elements of
        the recurrent matrix are randomly chosen based on rnn_portions.
    dropout : float
        Dropout rate between 0 and 1.
    use_normal_init : bool
        Whether to use a normal distribution for the neuron parameters tauu,
        tauw, a and b. Uses a uniform distribution if False.
    use_bias: bool
        Whether to use bias or not in feedforward connections.
    normalization : str
        Type of normalization. Every string different from batchnorm and
        layernorm will result in no normalization.
    tauu_lim, tauw_lim, a_lim, b_lim : float lists
        Range of allowed values for the AdLIF neuron parameters.
    threshold : float
        Spike threshold value, should be 1.0 but is kept as a variable.
    dt : float
        Timestep of the simulation in ms (tauu and tauw are also in ms).
    return_all_spikes : bool
        If True, the forward also returns spikes from all hidden layers.
    """

    def __init__(
        self,
        input_size,
        layer_sizes=[144, 144],
        sfa_portions=[0.3, 0.5],
        rnn_portions=[0.2, 0.3],
        ff_portions=[0.2, 0.3],
        use_lateral_rnn=False,
        dropout=0.0,
        use_normal_init=True,
        use_bias=True,
        normalization="batchnorm",
        tauu_lim=[3.0, 25.0],
        tauw_lim=[30.0, 500.0],
        a_lim=[0.0, 5.0],
        b_lim=[0.0, 1.25],
        threshold=1.0,
        dt=1.0,
        return_all_spikes=False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = layer_sizes[-1]
        self.num_layers = len(layer_sizes)
        self.return_all_spikes = return_all_spikes

        if len(rnn_portions) != self.num_layers:
            raise ValueError("rnn_portions must contain num_layers values.")

        if type(use_lateral_rnn) is bool:
            use_lateral_rnn = self.num_layers * [use_lateral_rnn]

        self.snn = nn.ModuleList([])
        self.unused_param_count = 0

        for i in range(self.num_layers):
            self.snn.append(
                SpikingFCLayer(
                    input_size=input_size if i == 0 else layer_sizes[i - 1],
                    output_size=layer_sizes[i],
                    sfa_portion=sfa_portions[i],
                    rnn_portion=rnn_portions[i],
                    ff_portion=ff_portions[i],
                    use_lateral_rnn=use_lateral_rnn[i],
                    use_normal_init=use_normal_init,
                    use_bias=use_bias,
                    normalization=normalization,
                    dropout=dropout,
                    tauu_lim=tauu_lim,
                    tauw_lim=tauw_lim,
                    a_lim=a_lim,
                    b_lim=b_lim,
                    threshold=threshold,
                    dt=dt,
                )
            )
            self.unused_param_count += self.snn[i].unused_param_count

    def forward(self, x):

        all_spikes = []
        for snn_lay in self.snn:
            x = snn_lay(x)
            all_spikes.append(x)

        if self.return_all_spikes:
            return x, all_spikes
        else:
            return x


class SpikingFCLayer(nn.Module):
    """
    Single layer of spiking neurons with fully connected feedforward synaptic
    weights. The neuron stimuli with shape (batch, time, output_size) is a
    linear combination of the input with shape (batch, time, input_size).
    """

    def __init__(
        self,
        input_size,
        output_size,
        sfa_portion=0.5,
        rnn_portion=0.3,
        ff_portion=0.3,
        use_lateral_rnn=False,
        use_normal_init=True,
        use_bias=True,
        normalization="batchnorm",
        dropout=0.0,
        tauu_lim=[3.0, 25.0],
        tauw_lim=[30.0, 500.0],
        a_lim=[0.0, 5.0],
        b_lim=[0.0, 1.25],
        threshold=1.0,
        dt=1.0,
        **kwargs,
    ):
        super().__init__()

        # Fixed parameters
        self.output_size = output_size
        self.ff_portion = ff_portion

        # Trainable parameters
        self.W = nn.Linear(input_size, output_size, bias=use_bias)

        self.W_mask = init_connection_mask(
            input_size=input_size,
            output_size=output_size,
            nonzero_portion=ff_portion,
            use_lateral=False,
            use_zero_diag=False,
        )

        # Normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(output_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(output_size)
            self.normalize = True

        # Spiking neurons
        self.spiking_neurons = SpikingNeurons(
            input_size=output_size,
            sfa_portion=sfa_portion,
            rnn_portion=rnn_portion,
            use_lateral_rnn=use_lateral_rnn,
            use_normal_init=use_normal_init,
            dropout=0.0,
            tauu_lim=tauu_lim,
            tauw_lim=tauw_lim,
            a_lim=a_lim,
            b_lim=b_lim,
            threshold=threshold,
            dt=dt,
        )

        # Dropout
        self.drop = nn.Dropout(p=dropout)

        # Count unused (masked) parameters
        self.unused_param_count = (1 - self.W_mask).sum().detach().cpu().numpy()
        self.unused_param_count += self.spiking_neurons.unused_param_count

    def forward(self, x):

        # Apply dropout to input (dropout is set to 0 in self.spiking_neurons)
        x = self.drop(x)

        # Mask portion of feeforward connections
        if self.ff_portion != 1.0:
            self.W.weight.data = self.W.weight * self.W_mask.to(x.device)

        # Apply linear operation on input to produce neuron stimuli
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Pass stimuli through spiking neurons
        spikes = self.spiking_neurons(Wx)

        return spikes


class SpikingNeurons(nn.Module):
    """
    Single spiking layer where the input with shape (batch, time, feats)
    is directly the neuron stimuli so that there is one neuron per input
    feature and the output size of the layer is the same as the input.
    """

    def __init__(
        self,
        input_size,
        sfa_portion=0.5,
        rnn_portion=0.01,
        use_lateral_rnn=True,
        use_normal_init=False,
        dropout=0.0,
        tauu_lim=[3, 25],
        tauw_lim=[30, 350],
        a_lim=[-0.5, 5.0],
        b_lim=[0.0, 2.0],
        threshold=1.0,
        dt=10.0,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = input_size
        self.use_normal_init = use_normal_init
        self.sfa_portion = sfa_portion
        self.rnn_portion = rnn_portion
        self.tauu_min, self.tauu_max = tauu_lim
        self.tauw_min, self.tauw_max = tauw_lim
        self.a_min, self.a_max = a_lim
        self.b_min, self.b_max = b_lim
        self.threshold = threshold
        self.dt = dt

        # Trainable recurrent matrix
        if rnn_portion > 0:
            self.V = nn.Linear(input_size, input_size, bias=False)
        else:
            self.V = ZeroTimes()

        # Trainable neuron parameters
        self.tauu = nn.Parameter(torch.Tensor(input_size))
        self.tauw = nn.Parameter(torch.Tensor(input_size))
        self.a = nn.Parameter(torch.Tensor(input_size))
        self.b = nn.Parameter(torch.Tensor(input_size))
        self.init_params()

        # Masks for SFA and recurrent connections
        self.sfa_mask = init_sfa_mask(
            hidden_size=input_size,
            sfa_portion=sfa_portion,
        )
        self.rnn_mask = init_connection_mask(
            input_size=input_size,
            output_size=input_size,
            nonzero_portion=rnn_portion,
            use_lateral=use_lateral_rnn,
            use_zero_diag=True,
        )
        self.unused_param_count = (1 - self.sfa_mask).sum().detach().cpu().numpy()
        if rnn_portion > 0:
            self.unused_param_count += (1 - self.rnn_mask).sum().detach().cpu().numpy()

        # Activation with surrogate gradient
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        self.device = x.device

        # Apply dropout to input
        x = self.drop(x)

        # Bounds on neuron parameters and masks
        self.clamp_neuron_parameters()
        self.apply_masks()

        # Compute output spike trains from neuron dynamics
        spikes = self._radlif_cell(x)

        return spikes

    def _radlif_cell(self, x):
        """
        Dynamics of the neurons membrane potential and spiking.
        """
        # Initializations
        device = x.device
        ut = torch.zeros(x.shape[0], x.shape[2]).to(device)
        wt = torch.zeros(x.shape[0], x.shape[2]).to(device)
        st = torch.zeros(x.shape[0], x.shape[2]).to(device)
        spikes = []

        # Compute AdLIF parameters from their SRM counterparts
        alpha = torch.exp(-self.dt / self.tauu)
        beta = torch.exp(-self.dt / self.tauw)

        # Neuron spiking dynamics
        for t in range(x.shape[1]):
            ut = alpha * (ut - st) - (1 - alpha) * wt + x[:, t] + self.V(st)
            wt = beta * (wt + self.b * st) + (1 - beta) * self.a * ut
            st = self.spike_fct(ut - self.threshold)
            spikes.append(st)

        return torch.stack(spikes, dim=1)

    def clamp_neuron_parameters(self):
        """
        Bound the values of the neuron trainable parameters.
        An upper bound is defined for 'a' so that eigenvalues are real.
        """
        self.tauu.data.clamp_(self.tauu_min, self.tauu_max)
        self.tauw.data.clamp_(self.tauw_min, self.tauw_max)
        self.b.data.clamp_(self.b_min, self.b_max)
        a_min = self.a_min * torch.ones(self.input_size).to(self.device)
        a_max = (self.tauw - self.tauu) ** 2 / (4 * self.tauu * self.tauw) - 1e-5
        a_max = torch.minimum(a_max, torch.Tensor([self.a_max]).to(self.tauu.device))
        self.a.data.clamp_(a_min, a_max)

    def apply_masks(self):
        """
        Applies a fixed binary mask to SFA parameters among neurons so that ones
        are AdLIF and zeros LIF. Similarly it applies a mask to the recurrent
        matrix elements to reduce the number of connections. Diagonal elements
        are all zeros to retain refractoriness, which is handled separately.
        """
        self.a.data = self.a * self.sfa_mask.to(self.device)
        self.b.data = self.b * self.sfa_mask.to(self.device)
        if self.rnn_portion > 0:
            self.V.weight.data = self.V.weight * self.rnn_mask.to(self.device)
            self.V.weight.data.fill_diagonal_(0)

    def init_params(self):
        """
        This initializes the neuron parameters tauu, tauw, a and b
        within physiologically plausible ranges, either with normal
        or uniform distributions.
        """
        if self.use_normal_init:
            tauu_mean = 0.5 * (self.tauu_min + self.tauu_max)
            tauu_std = (self.tauu_max - tauu_mean) / 3  # 99.3%
            tauw_mean = 0.5 * (self.tauw_min + self.tauw_max)
            tauw_std = (self.tauw_max - tauw_mean) / 3
            a_mean = 0.5 * (self.a_min + self.a_max)
            a_std = (self.a_max - a_mean) / 3
            b_mean = 0.5 * (self.b_min + self.b_max)
            b_std = (self.b_max - b_mean) / 3
            nn.init.normal_(self.tauu, tauu_mean, tauu_std)
            nn.init.normal_(self.tauw, tauw_mean, tauw_std)
            nn.init.normal_(self.a, a_mean, a_std)
            nn.init.normal_(self.b, b_mean, b_std)
        else:
            nn.init.uniform_(self.tauu, self.tauu_min, self.tauu_max)
            nn.init.uniform_(self.tauw, self.tauw_min, self.tauw_max)
            nn.init.uniform_(self.a, self.a_min, self.a_max)
            nn.init.uniform_(self.b, self.b_min, self.b_max)


class ReadoutLayer(nn.Module):
    """
    This function implements a single layer of non-spiking Leaky
    Integrate-and-Fire (LIF) neurons, where the output consists of
    a cumulative sum of the membrane potential using a softmax
    function, instead of spikes.

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    hidden_size : int
        Number of output neurons.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=True,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute membrane potential via non-spiking neuron dynamics
        out = self._readout_cell(Wx)

        return out

    def _readout_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        out = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute potential (LIF)
            ut = alpha * ut + (1 - alpha) * Wx[:, t, :]
            out = out + F.softmax(ut, dim=1)

        # Add time dim (for loss fct)
        out = out.unsqueeze(dim=1)

        return out


if __name__ == "__main__":

    # Generate random input
    batch_size, num_steps, input_size = 8, 500, 100
    x = (torch.rand(batch_size, num_steps, input_size) > 0.9).float()

    # Instantiate modules
    auditory_neurons = SpikingNeurons(
        input_size=input_size,
        sfa_portion=0.0,
        rnn_portion=0.0,
        use_lateral_rnn=False,
        dropout=0.0,
        dt=1.0,
    )
    net = MultilayeredSpiking(
        input_size=input_size,
        layer_sizes=[144, 144, 144],
        sfa_portions=[0.3, 0.5, 0.7],
        rnn_portions=[0.3, 0.5, 0.7],
        ff_portions=[0.3, 0.5, 0.7],
        use_lateral_rnn=False,
        dropout=0.1,
        dt=1.0,
        return_all_spikes=True,
    )

    # Pass input through module
    y_auditory = auditory_neurons(x)
    y, all_spikes = net(y_auditory)

    print(y_auditory.shape, y.shape)
