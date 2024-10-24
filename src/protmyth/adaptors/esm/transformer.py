# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Esm Transformers mapper
"""

from protmyth.adaptors.mapper import ModuleMapper
from protmyth.adaptors.common import linear_mapper, layer_norm_mapper
from protmyth.adaptors.esm.layers import attention_mapper


def transformer_mapper(tgt_pfx: str = "", src_pfx: str = "") -> ModuleMapper:
    """Creates a module mapper for transformer layers.

    Parameters
    ----------
    tgt_pfx : str, optional
        Target prefix for module mapping, by default "".
    src_pfx : str, optional
        Source prefix for module mapping, by default "".

    Returns
    -------
    ModuleMapper
        A configured ModuleMapper object for transformer layers.
    """

    return (
        ModuleMapper(tgt_pfx, src_pfx)
        .add_submodule(attention_mapper("attention", "self_attn"))
        .add_submodule(layer_norm_mapper("self_attn_layer_norm", "self_attn_layer_norm"))
        .add_submodule(linear_mapper("fc1", "fc1"))
        .add_submodule(linear_mapper("fc2", "fc2"))
        .add_submodule(layer_norm_mapper("final_layer_norm", "final_layer_norm"))
    )
