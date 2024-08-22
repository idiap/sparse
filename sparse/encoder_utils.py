#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
"""
This is where utilitary modules for the encoders are defined.
"""
import torch.nn as nn


def print_num_params(module, name):
    """
    This is to print the number of trainable parameters in a module.
    """
    n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if hasattr(module, "unused_param_count"):
        n_params -= module.unused_param_count
    if n_params < 1e3:
        print(f"\n{name} has {int(n_params)} trainable params.\n")
    elif n_params < 1e6:
        print(f"\n{name} has {n_params / 1e3:.1f}k trainable params.\n")
    else:
        print(f"\n{name} has {n_params / 1e6:.1f}M trainable params.\n")


class Permute(nn.Module):
    """
    This is to permute the dimensions of a tensor.
    """

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.permute(self.shape)


class MergeFeatsAndChannels(nn.Module):
    """
    This reshapes conv output from (batch, steps, feats, channels) to
    (bath, steps, feats * channels).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch, steps, feats, channels = x.size()
        return x.contiguous().view(batch, steps, feats * channels)
