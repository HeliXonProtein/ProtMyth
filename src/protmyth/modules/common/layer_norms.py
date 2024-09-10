# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""
Layer Norm in common modules
"""

from torch import nn


class LayerNorm(nn.Module):
    """
    LayerNorm
    """

    def __init__(self) -> None:
        super().__init__()
