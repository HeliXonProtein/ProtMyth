import torch
import torch.nn as nn
from graphviz import Digraph
from jaxtyping import Float, Shaped
from einops import rearrange
from protmyth.modules.base import BaseModule
from typing import Sequence
import torchviz

class ESM1LayerNorm(BaseModule):
    """
    Layer normalization for ESM1 model.

    Parameters
    ----------
    hidden_size : int or Sequence[int]
        Size of the hidden layer.
    eps : float, optional
        A small value added to variance to avoid division by zero (default is 1e-12).
    affine : bool, optional
        If True, the module has learnable affine parameters (default is True).
    """
    
    def __init__(self, hidden_size: int | Sequence[int], eps: float = 1e-12, affine: bool = True):
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.hidden_size))
            self.bias = nn.Parameter(torch.zeros(self.hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x: Float[torch.Tensor, "batch ..."]) -> Float[torch.Tensor, "batch ..."]:
        """
        Perform forward pass of the layer normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ... , hidden_size).

        Returns
        -------
        torch.Tensor
            Normalized output tensor.
        """
        # Compute mean and variance across specified dimensions
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        
        if self.affine:
            x = (self.weight * x) + self.bias
        return x

    def make_graph(self, batch_dims: Sequence[int], device: torch.device) -> Digraph:
        """
        Create a graph representation of the layer norm operation.

        Parameters
        ----------
        batch_dims : Sequence[int]
            Dimensions of the batch.
        device : torch.device
            The device on which the tensor is located.

        Returns
        -------
        Digraph
            A graph representing the forward pass.
        """
        x_data = torch.randn(list(batch_dims) + list(self.hidden_size), device=device)
        output = self.forward(x_data)
        return torchviz.make_dot(output.mean(), params=dict(self.named_parameters()))


class ESM1bLayerNorm(BaseModule):
    """
    Layer normalization for ESM1b model.

    Parameters
    ----------
    hidden_size : int or Sequence[int]
        Size of the hidden layer.
    eps : float, optional
        A small value added to variance to avoid division by zero (default is 1e-12).
    affine : bool, optional
        If True, the module has learnable affine parameters (default is True).
    """

    def __init__(self, hidden_size: int | Sequence[int], eps: float = 1e-12, affine: bool = True):
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.hidden_size))
            self.bias = nn.Parameter(torch.zeros(self.hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x: Float[torch.Tensor, "batch ..."]) -> Float[torch.Tensor, "batch ..."]:
        """
        Perform forward pass of the layer normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ... , hidden_size).

        Returns
        -------
        torch.Tensor
            Normalized output tensor.
        """
        # Compute mean and variance across specified dimensions
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        
        if self.affine:
            x = (self.weight * x) + self.bias
        return x

    def make_graph(self, batch_dims: Sequence[int], device: torch.device) -> Digraph:
        """
        Create a graph representation of the layer norm operation.

        Parameters
        ----------
        batch_dims : Sequence[int]
            Dimensions of the batch.
        device : torch.device
            The device on which the tensor is located.

        Returns
        -------
        Digraph
            A graph representing the forward pass.
        """
        x_data = torch.randn(list(batch_dims) + list(self.hidden_size), device=device)
        output = self.forward(x_data)
        return torchviz.make_dot(output.mean(), params=dict(self.named_parameters()))

