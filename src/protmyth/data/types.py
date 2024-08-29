# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""This module contains the data types used in ProtMyth.
"""

from typing import TypedDict

import torch


# Protein sequence classes
class ProteinSequence(TypedDict):
    """A dictionary containing the protein sequence and its length.

    Parameters
    ----------
    seq : torch.Tensor
        The protein sequence represented as a tensor.
    seq_len : int
        The length of the protein sequence.
    """
    seq: torch.Tensor
    seq_len: int


class ProteinSequenceDomain(TypedDict):
    """A dictionary containing the definition of protein sequence domain.

    Parameters
    ----------
    alphabet : list[str]
        The set of characters representing amino acids in the protein sequence.
    mapping : dict[str, torch.Tensor]
        A dictionary mapping each amino acid character to its corresponding representation.

    Notes
    -----
    This class defines the domain of protein sequences by specifying the alphabet
    of amino acids and their corresponding numerical representations. For different training tasks,
    there maybe slightly different for using '-', 'X' and other combined label.
    """
    alphabet: list[str]
    mapping: dict[str, torch.Tensor]


class ProteinSequenceTensor(TypedDict):
    """A dictionary containing the protein sequence tensor for LM input.

    Parameters
    ----------
    seq_input : torch.Tensor
        The protein sequence represented as a tensor of integers.
    seq_chain : torch.IntTensor
        The chain identifiers for each residue in the sequence.
    seq_mask : torch.BoolTensor
        A mask tensor indicating valid positions in the sequence.
    positions : torch.Tensor
        The position indices for each residue in the sequence.
    repr_left_shift : int
        The number of positions to shift the representation to the left.
    repr_right_shift : int
        The number of positions to shift the representation to the right.

    Notes
    -----
    This class represents the input tensors required for language model processing
    of protein sequences. It includes the sequence itself, chain information,
    masking, positional information, and shift parameters for representation.
    """
    seq_input: torch.Tensor
    seq_chain: torch.IntTensor
    seq_mask: torch.BoolTensor
    positions: torch.Tensor
    repr_left_shift: int
    repr_right_shift: int
