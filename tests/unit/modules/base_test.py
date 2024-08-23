# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Unit tests for the BaseModule class.
"""

from unittest import TestCase
from unittest.mock import patch

import torch
from graphviz.graphs import Digraph
from jaxtyping import Float

from protmyth.modules.base import BaseModule


_ForwardReturnType = Float[torch.Tensor, "..."]


class ConcreteModule(BaseModule[_ForwardReturnType]):
    """Helper class for testing the BaseModule class.

    Args:
        BaseModule (torch.Tensor): ...
    """

    def make_graph(self, *args, **kwargs) -> Digraph:
        return Digraph()


class TestBaseModule(TestCase):
    """Test suite for the BaseModule class.
    """

    def test_init(self):
        """Test the __init__ method.
        """
        module = ConcreteModule()
        self.assertIsInstance(module, BaseModule)

    @patch("protmyth.modules.base.Digraph")
    def test_make_graph(self, mock_digraph):
        """Test the make_graph method.
        """
        module = ConcreteModule()
        module.make_graph()
