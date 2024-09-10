# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""
Axial Attention Stacks from ESM
"""

from torch import nn


class Transformer(nn.Module):
    """
    Transformer
    """

    def __init__(self) -> None:
        super().__init__()
