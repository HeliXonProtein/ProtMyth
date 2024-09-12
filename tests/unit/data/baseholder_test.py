# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Unit test for the BaseDataHolder class.
"""

from collections.abc import Iterable
from typing import Any
from unittest import TestCase
from unittest.mock import patch

from protmyth.data.baseholder import BaseHolder
from protmyth.data.amino_acid_constants import (
    STD_AA_WITHUNK_Domain,
)
from protmyth.data.types import (
    ProteinSequence,
    ProteinSequenceDomain,
    ProteinSequenceTensor,
)


class ConcreteHolder(BaseHolder[ProteinSequence, ProteinSequenceTensor, ProteinSequenceDomain]):
    """Helper class for testing the BaseHolder class.

    Parameters
    ----------
    BaseHolder : ProteinSequence, ProteinSequenceTensor, ProteinSequenceDomain
        The base holder class to be tested.
        This class should have at least three type variables:
        ProteinSequence, ProteinSequenceTensor, and ProteinSequenceDomain.
        This class should also have three methods:
        read_posgres_raw: read data from a PostgreSQL database
        transform_raw_to_data: transform raw data to ProteinSequence
        yield_raw_from_file: yield raw data from a file (e.g. FASTA)
    """
    @classmethod
    def read_postgres_raw(cls, input_x: Any, domain: ProteinSequenceDomain, **kwargs) -> ProteinSequence:
        """mock read_postgres_raw method

        Parameters
        ----------
        input_x : Any
            mock input
        domain : ProteinSequenceDomain
            example domain
        **kwargs : Any
            additional arguments

        Returns
        -------
        ProteinSequence
            mock return value
        """
        return ProteinSequence(seq='ABC', seq_len=3)

    @classmethod
    def yield_raw_from_file(cls, file_path: str, **kwargs) -> Iterable[ProteinSequence]:
        """mock yield_raw_from_file method

        Parameters
        ----------
        file_path : str
            mock input file path
        **kwargs : Any
            additional arguments

        Yields
        ------
        ProteinSequence
            mock yield value
        """
        yield ProteinSequence(seq='ABC', seq_len=3)


class TestBaseHolder(TestCase):
    """Test the BaseHolder class.
    """
    def test_abstract_methods(self):
        """Test the abstract methods.
        """
        with self.assertRaises(TypeError):
            holder = ConcreteHolder()  # type: ignore
            for _data in holder.yield_raw_from_file('example'):
                proteinsequence = _data
                break
            with self.assertRaises(NotImplementedError):
                holder.transform_raw_to_data(proteinsequence, STD_AA_WITHUNK_Domain)
