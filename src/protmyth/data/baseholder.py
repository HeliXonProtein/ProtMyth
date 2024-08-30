# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Base holder class.
"""

import abc
import dataclasses
from typing import Generic, TypeVar


_RawT = TypeVar("_RawT")
_LabelT = TypeVar("_LabelT")


@dataclasses.dataclass
class BaseHolder(Generic[_RawT, _LabelT], metadata=abc.ABCMeta):
    """Base class for all data holder

    This class serves as a foundation for all data holders in the system.
    It defines abstract methods that must be implemented by subclasses to handle
    raw data reading, transformation, and file operations.

    Parameters
    ----------
    _RawT : TypeVar
        Type variable for raw data.
    _LabelT : TypeVar
        Type variable for label data.
    """

    @abc.abstractmethod
    def read_raw(self, *args, **kwargs) -> _RawT:
        """Transform a python object read from database into raw datatype.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        _RawT
            Raw data type predefined.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("read_raw method not implemented")

    @abc.abstractmethod
    def transform_raw_to_label(self, raw: _RawT, *args, **kwargs) -> _LabelT:
        """Transform raw data to label data.

        Parameters
        ----------
        raw : _RawT
            Raw data type predefined.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        _LabelT
            Label data type predefined.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("transform_raw_to_label method not implemented")

    @abc.abstractmethod
    def read_raw_from_file(self, file_path: str, *args, **kwargs) -> _RawT:
        """Read raw data from a file.

        Parameters
        ----------
        file_path : str
            Path to the file containing raw data.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        _RawT
            Raw data type predefined.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("read_raw_from_file method not implemented")
