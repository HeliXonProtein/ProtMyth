# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""
This module contains the base class for all ProtMyth models.
"""

import abc
from typing import Generic, TypeVar, Literal, Any, overload

import lightning as L


_InputType = TypeVar("_InputType")
_ResultType = TypeVar("_ResultType")
_LossType = TypeVar("_LossType")


class BaseModel(L.LightningModule, Generic[_InputType, _ResultType, _LossType], metaclass=abc.ABCMeta):
    """
    Base class for all ProtMyth models.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @overload
    def __call__(self, batch: _InputType, get_loss: Literal[False], *args: Any, **kwargs: Any) -> _ResultType:
        ...

    @overload
    def __call__(
        self, batch: _InputType, get_loss: Literal[True], *args: Any, **kwargs: Any
    ) -> tuple[_ResultType, _LossType]:
        ...

    def __call__(
        self, batch: _InputType, get_loss: Literal[False, True], *args: Any, **kwargs: Any
    ) -> _ResultType | tuple[_ResultType, _LossType]:
        return super().__call__(batch, get_loss, **kwargs)
