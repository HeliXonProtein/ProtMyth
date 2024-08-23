# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""This module contains the base class for all ProtMyth models.
"""


import abc
from typing import Generic, TypeVar, Literal, Any, overload

import lightning as L


_InputT = TypeVar("_InputT")
_ResultT = TypeVar("_ResultT")
_LossT = TypeVar("_LossT")


class BaseModel(L.LightningModule, Generic[_InputT, _ResultT, _LossT], metaclass=abc.ABCMeta):
    """Base class for all ProtMyth models.
    """

    @overload
    def __call__(self, batch: _InputT, get_loss: Literal[False], *args: Any, **kwargs: Any) -> _ResultT:
        ...

    @overload
    def __call__(
        self, batch: _InputT, get_loss: Literal[True], *args: Any, **kwargs: Any
    ) -> tuple[_ResultT, _LossT]:
        ...

    def __call__(
        self, batch: _InputT, get_loss: Literal[False, True], *args: Any, **kwargs: Any
    ) -> _ResultT | tuple[_ResultT, _LossT]:
        return super().__call__(batch, get_loss, **kwargs)
