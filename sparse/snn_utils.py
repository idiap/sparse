#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
"""
This is where utilitary modules for the SNN are defined.
"""
import torch
import torch.nn as nn


class SpikeFunctionBoxcar(torch.autograd.Function):
    """
    Spike step function with box-car function as surrogate gradient.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= -0.5] = 0
        grad_x[x > 0.5] = 0
        return grad_x


def init_connection_mask(
    input_size,
    output_size,
    nonzero_portion,
    use_lateral=False,
    use_zero_diag=False,
):
    """
    Defines a binary matrix of shape (hidden_size, hidden_size) to mask
    a weight matrix. If use_lateral is True, the only nonzero elements
    are along the offset diagonals with offsets between -radius and +radius,
    where radius = nonzero_portion * hidden_size / 2. This means only nearby
    neurons within a spatial radius interact.
    If use_lateral is False, a random selection of connections is kept nonzero
    with no spatial constraint except that the diagonal remains zero.
    """
    if input_size == output_size:
        if use_lateral:
            radius = int(nonzero_portion * input_size / 2)
            mask = torch.zeros(input_size, input_size)
            for offset in range(-radius, radius + 1):
                mask = mask + torch.diag(
                    torch.ones(input_size - abs(offset)), diagonal=offset
                )
        else:
            _num_ones = int(nonzero_portion * input_size**2)
            _num_zeros = int(input_size**2 - _num_ones)
            mask = torch.cat((torch.zeros(_num_zeros), torch.ones(_num_ones)))
            mask = mask[torch.randperm(mask.numel())]
            mask = mask.view(input_size, input_size)
    else:
        mask = (torch.rand(output_size, input_size) < nonzero_portion).float()

        if use_lateral:
            raise NotImplementedError

    if use_zero_diag:
        mask.fill_diagonal_(0)

    return mask


def init_sfa_mask(hidden_size, sfa_portion):
    """
    This generates a binary mask for spike frequency adaptation.
    """
    num_ones = int(sfa_portion * hidden_size)
    num_zeros = int(hidden_size - num_ones)
    sfa_mask = torch.cat((torch.zeros(num_zeros), torch.ones(num_ones)))
    sfa_mask = sfa_mask[torch.randperm(sfa_mask.numel())]
    return sfa_mask


class ZeroTimes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.0
