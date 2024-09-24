# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

from protmyth.adaptors import mapper


def linear_mapper(target_prefix: str = "", source_prefix: str = "") -> mapper.ModuleMapper:
    """Creates a linear mapper for weight and bias.

    Parameters
    ----------
    target_prefix : str, optional
        The prefix for target keys (default is an empty string).
    source_prefix : str, optional
        The prefix for source keys (default is an empty string).

    Returns
    -------
    mapper.ModuleMapper
        A ModuleMapper instance with mappings for 'weight' and 'bias'.
    """
    return (
        mapper.ModuleMapper(target_prefix, source_prefix)
        .add("weight", "weight")
        .add("bias", "bias")
    )

def layer_norm_mapper(tgt_pfx: str = "", src_pfx: str = "") -> mapper.ModuleMapper:
    """Creates a layer normalization mapper for weight and bias.

    Parameters
    ----------
    tgt_pfx : str, optional
        The prefix for target keys (default is an empty string).
    src_pfx : str, optional
        The prefix for source keys (default is an empty string).

    Returns
    -------
    mapper.ModuleMapper
        A ModuleMapper instance with mappings for 'weight' and 'bias'.
    """
    return (
        mapper.ModuleMapper(tgt_pfx, src_pfx)
        .add("weight", "weight")
        .add("bias", "bias")
    )