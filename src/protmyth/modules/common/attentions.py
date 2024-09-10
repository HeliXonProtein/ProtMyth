# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""This module contains attention mechanisms used in ProtMyth. Based on the attention mechanism used,
we divide the attention mechanism into three categories: (https://arxiv.org/pdf/2203.14263)

1. Feature-Related:
    a. Multiplicity.
    b. Levels.
    c. Representations.
2. General:
    a. Scoring.
    b. Alignment.
    c. Dimensionality.
3. Query-Related:
    a. Type.
    b. Multiplicity.

We will implement the following attention mechanisms and transform them into protein based modules:

1. Place holder.
"""

from torch import nn


class Attention(nn.Module):
    """Attention
    """

    def __init__(self) -> None:
        super().__init__()
