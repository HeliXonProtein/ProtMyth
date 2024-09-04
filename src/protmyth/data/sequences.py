# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""This module contains sequences data holder for any sequence data.
"""

import gzip
import torch
from Bio.SeqIO import FastaIO
from collections.abc import Iterable
from typing import Any

from protmyth.data.baseholder import BaseHolder
from protmyth.data.amino_acid_constants import (
    STD_AA_WITHUNK_Domain,
)
from protmyth.data.types import (
    ProteinSequence,
    ProteinSequenceDomain,
    ProteinSequenceTensor,
)


class ProteinSequenceDataHolder(BaseHolder[ProteinSequence, ProteinSequenceTensor, ProteinSequenceDomain]):
    """data holder for any predefined protein sequence data types.
    """
    @classmethod
    def read_postgres_raw(
        cls,
        input_x: Any,
        domain: ProteinSequenceDomain = STD_AA_WITHUNK_Domain,
        **kwargs,
    ) -> ProteinSequence:
        raise NotImplementedError("Reading from postgres is not supported yet.")

    @classmethod
    def transform_raw_to_data(
        cls,
        raw: ProteinSequence | list[ProteinSequence],
        domain: ProteinSequenceDomain = STD_AA_WITHUNK_Domain,
        **kwargs,
    ) -> ProteinSequenceTensor:
        """Transform one or more protein sequences into a training
        data point for the model.

        Parameters
        ----------
        raw : ProteinSequence | list[ProteinSequence]
            Protein sequence(s) to be transformed.
        domain : ProteinSequenceDomain
            transformation domain.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        ProteinSequenceTensor
            protein data point in tensor format for model training.

        Raises
        ------
        ValueError
            raises if the input sequence contains invalid amino acids
            out of the specified domain.
        """
        if not isinstance(raw, list):
            raw = [raw]
        seq_input = []
        seq_chain = []
        seq_mask = []
        positions = []
        repr_left_shift = 0
        repr_right_shift = 0
        for seqidx, sequence in enumerate(raw):
            for aa_pos, aa in enumerate(sequence["seq"]):
                if aa not in domain["alphabet"]:
                    raise ValueError(f"Invalid amino acid: {aa} in {sequence['seq']}")
                else:
                    seq_input.append(domain["mapping"][aa])
                    seq_chain.append(seqidx)
                    # TODO: add masking strategy here
                    seq_mask.append(0)
                    positions.append(aa_pos)

        return ProteinSequenceTensor(
            seq_input=torch.Tensor(seq_input),
            seq_chain=torch.IntTensor(seq_chain),
            seq_mask=torch.BoolTensor(seq_mask),
            positions=torch.Tensor(positions),
            repr_left_shift=repr_left_shift,
            repr_right_shift=repr_right_shift,
        )

    @classmethod
    def yield_raw_from_file(
        cls,
        file_path: str,
        **kwargs,
    ) -> Iterable[ProteinSequence]:
        """Yield protein sequences from a file.

        Parameters
        ----------
        file_path : str
            path to the file to read from, including:
            1. fasta file.
            2. gzipped fasta file.
        **kwargs
            Arbitrary keyword arguments.

        Yields
        ------
        Iterator[Iterable[ProteinSequence]]
            yield one protein sequence at a time.

        Raises
        ------
        ValueError
            If the file format is not supported.
        """
        match file_path:
            case "*.fasta" | "*.fa":
                with open(file_path, encoding="utf-8") as handle:
                    for record in FastaIO.SimpleFastaParser(handle):
                        yield ProteinSequence(
                            seq=record[-1],
                            seq_len=len(record[-1]),
                        )
            case "*.fasta.gz" | "*.fa.gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as handle:
                    for record in FastaIO.SimpleFastaParser(handle):
                        yield ProteinSequence(
                            seq=record[-1],
                            seq_len=len(record[-1]),
                        )
            case _:
                raise ValueError(f"Unsupported file format: {file_path}")
