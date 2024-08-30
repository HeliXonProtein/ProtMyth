# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""This module contains sequences data holder for any sequence data.
"""

import dataclasses
import torch
import jaxtyping

from protmyth.data import types
from protmyth.baseholder import BaseHolder





class ProteinSequenceDataHolder(BaseHolder):
    """data holder for any predefined protein sequence data types.
    """
    