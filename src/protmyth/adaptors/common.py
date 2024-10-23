# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""  ProtMyth Adaptors Module

This module provides functionality for creating mappers that facilitate the transformation
and alignment of model parameters between different neural network architectures. It is
part of the ProtMyth library, which is designed to streamline the process of adapting
pre-trained models to new tasks or architectures.

The module includes:

- `linear_mapper`: A function to create a mapper for linear layers, mapping 'weight' and 'bias'.
- `layer_norm_mapper`: A function to create a mapper for layer normalization layers, mapping 'weight' and 'bias'.

Each mapper is an instance of `ModuleMapper`, which allows for flexible and customizable
mapping of model parameters using specified prefixes for target and source keys.
"""


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
