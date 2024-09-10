# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Utils for auxilary losses like bert loss
"""

import torch
from torch import nn
import torch.nn.functional as F

import numbers
import collections

def mask_mean(mask, value, dim=None, drop_mask_channel=False, eps=1e-10):
    """Masked mean."""
    axis = dim
    if drop_mask_channel:
        mask = mask[..., 0]

    mask_shape = mask.shape
    value_shape = value.shape

    assert len(mask_shape) == len(value_shape)

    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(mask_shape)))
    assert isinstance(axis, collections.Iterable), (
        'axis needs to be either an iterable, integer or "None"')

    broadcast_factor = 1.
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            assert mask_size == value_size

    return (torch.sum(mask * value, dim=axis) /
            (torch.sum(mask, dim=axis) * broadcast_factor + eps))