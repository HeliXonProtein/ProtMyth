# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""This module contains sequences data holder for any sequence data.
"""

import gzip
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


class ProteinSequenceDataHolder(BaseHolder[ProteinSequence, ProteinSequenceTensor]):
    """data holder for any predefined protein sequence data types.
    """
    @classmethod
    def read_raw(
        cls,
        input_x: Any,
        domain: ProteinSequenceDomain = STD_AA_WITHUNK_Domain,
        **kwargs,
    ) -> ProteinSequence:
        pass

    @classmethod
    def transform_raw_to_data(
        cls,
        raw: ProteinSequence,
        domain: ProteinSequenceDomain,
        **kwargs,
    ) -> ProteinSequenceTensor:
        pass

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
                            record[-1],
                            len(record[-1]),
                        )
            case "*.fasta.gz" | "*.fa.gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as handle:
                    for record in FastaIO.SimpleFastaParser(handle):
                        yield ProteinSequence(
                            record[-1],
                            len(record[-1]),
                        )
            case _:
                raise ValueError(f"Unsupported file format: {file_path}")
