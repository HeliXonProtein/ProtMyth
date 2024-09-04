# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Base holder class.
"""

import abc
import dataclasses
from typing import Generic, TypeVar, Any
from collections.abc import Iterable


_RawT = TypeVar("_RawT")
_LabelT = TypeVar("_LabelT")
_DomainT = TypeVar("_DomainT")


@dataclasses.dataclass
class BaseHolder(Generic[_RawT, _LabelT, _DomainT], metaclass=abc.ABCMeta):
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
    _DomainT : TypeVar
        Type variable for domain data.
    """

    @classmethod
    @abc.abstractmethod
    def read_postgres_raw(cls, input_x: Any, domain: _DomainT, **kwargs) -> _RawT:
        """Transform a python object read from postgresql database into raw datatype.

        Parameters
        ----------
        input_x : Any
            Input object.
        domain : _DomainT
            Domain type predefined.
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

    @classmethod
    @abc.abstractmethod
    def transform_raw_to_data(cls, raw: _RawT | list[_RawT], domain: _DomainT, **kwargs) -> _LabelT:
        """Transform raw data to training data.

        Parameters
        ----------
        raw : _RawT | list[_RawT]
            Raw data type predefined. If a list is provided, it will be transformed
            into a single data point.
        domain : _DomainT
            Domain type predefined.
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

    @classmethod
    @abc.abstractmethod
    def yield_raw_from_file(cls, file_path: str, **kwargs) -> Iterable[_RawT]:
        """Yield raw data from a file.

        Parameters
        ----------
        file_path : str
            Path to the file containing raw data.
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
