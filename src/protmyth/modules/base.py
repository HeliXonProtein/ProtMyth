# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""This module contains the base class for all ProtMyth modules.
"""


import abc
from typing import Generic, TypeVar, Any

from graphviz import Digraph
# import jaxtyping
from torch import nn


# NOTE: jaxtyping dtype[array, shape] will be treated as array itself in mypy check
# _ModuleReturnT = TypeVar("_ModuleReturnT", bound=jaxtyping.AbstractArray)
_ModuleReturnT = TypeVar("_ModuleReturnT")


class BaseModule(nn.Module, Generic[_ModuleReturnT]):
    """Base class for all ProtMyth modules.

    Args:
        Generic (_type_): Type variable for the forward return type.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs) -> _ModuleReturnT:
        """Forward pass of the module with a specified return type.

        Args:
            *args: Positional arguments to be passed to the module's forward method.
            **kwargs: Keyword arguments to be passed to the module's forward method.

        Returns:
            _ModuleReturnT: return type of the module's forward method.
        """
        return super().forward(*args, **kwargs)

    def __call__(self, *args: Any, **kwds: Any) -> _ModuleReturnT:
        return super().__call__(*args, **kwds)

    @abc.abstractmethod
    def make_graph(self, *args, **kwargs) -> Digraph:
        """Generate a graphviz Digraph object representing the module's computation graph.

        Args:
            *args: Positional arguments to be passed to the module's forward method.
            **kwargs: Keyword arguments to be passed to the module's forward method.

        Returns:
            A graphviz Digraph object representing the module's computation graph.
        """
        raise NotImplementedError("make_graph method not implemented")
