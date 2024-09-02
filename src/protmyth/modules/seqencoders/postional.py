# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Positional sequence encodere modules.
"""

from collections.abc import Sequence

import einops
import torch
import jaxtyping
import torchviz
from graphviz import Digraph

from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module


@register_module("seqencoders")
class RotatedPostionalEncoder(BaseModule[jaxtyping.Array[torch.Tensor, "..."]]):
    r"""Rotated Positional Encoding (RoPE) for Transformers
    This is a PyTorch implementation of the RoPE layer described in
    `Enhanced Transformer with Rotary Position Embedding`_.
    (https://arxiv.org/abs/2104.09864)

    This is an N-dimensional implementation of the RoPE layer, which can be
    used for multidimensional positions, like the x-y axis of an image.
    """
    _rope_cache: dict[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] = {}

    def __init__(
        self,
        kv_channels: int,
        position_dimensions: int,
        frequencies: Sequence[float] | float = 1e4,
        repo_chache_maximum: int = 10,
    ) -> None:
        """initionalize the RoPE layer.

        Args:
            kv_channels (int): The size of the input tensor.
            position_dimensions (int): The dimension of the positional encoding.
            frequencies (Sequence[float] | float, optional):
            The frequencies of the RoPE positional encoding. Defaults to 1e4.
            repo_chache_maximum (int, optional): Ths maximum size of RoPE cache. Defaults to 10.
        """
        super().__init__()
        self.kv_channels = kv_channels
        self.position_dimensions = position_dimensions
        self.frequencies = frequencies
        self.repo_chache_maximum = repo_chache_maximum

        if self.kv_channels % self.position_dimensions:
            raise ValueError(
                f"kv_channels ({kv_channels}) must be divisible"
                f" by position_dimensions ({position_dimensions})"
            )
        self.half_size = int(kv_channels // 2 // position_dimensions)
        if not self.half_size:
            raise ValueError(
                f"RoPE dimension {position_dimensions} must be less than"
                f" half of kv_channels {kv_channels}"
            )
        if self.position_dimensions < 1:
            raise ValueError(f"{position_dimensions} must be greater than 1")
        self.freqs = self._expand_freqs(frequencies)

        # Initialize the tensor buffer for the RoPE cache
        # NOTE: Let the lightning module handle device and dtype
        half_range = torch.arange(self.half_size).div(self.half_size)
        _freqs = torch.tensor(self.freqs)[..., None]
        self.register_buffer("pre_sin", torch.pow(_freqs, -half_range), persistent=False)

    def _expand_freqs(self, freqs: Sequence[float] | float) -> Sequence[float]:
        if isinstance(freqs, Sequence):
            if len(freqs) != self.position_dimensions:
                raise ValueError(
                    f"Length of frequencies {len(freqs)} must be equal to"
                    f" position_dimensions {self.position_dimensions}"
                )
            return freqs
        else:
            return [freqs] * self.position_dimensions

    def get_sin_cos(
        self,
        positions: jaxtyping.Array[torch.Tensor, "... N D"],
    ) -> tuple[jaxtyping.Array[torch.Tensor, "... N H C"], jaxtyping.Array[torch.Tensor, "... N H C"]]:
        """Get the sine and cosine of the RoPE.

        Args:
            positions (torch.Tensor): The positions tensor of shape (..., N, D).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The sine and cosine of the RoPE.

        Raises:
            ValueError: If the position_dimensions is not 1 and the last dimension of the
            positions is not equal to position_dimensions.
        """
        if positions.shape[-1] != self.position_dimensions:
            if self.position_dimensions != 1:
                raise ValueError(
                    f"Expected {self.position_dimensions} but got {positions.shape[-1]}"
                )
            positions = einops.repeat(positions, "... N ->... N 1")
        positions = einops.repeat(positions, "... D -> ... 1 D 1")
        sinusoid = self.pre_sin * positions
        sin, cos = torch.sin(sinusoid), torch.cos(sinusoid)
        return sin, cos

    def trunk_half(
        self,
        x: jaxtyping.Array[torch.Tensor, "... N H C"]
    ) -> jaxtyping.Array[torch.Tensor, "... N H C"]:
        """Trunk the input tensor to half of its size and reconcate it.

        Args:
            x (torch.Tensor): The input tensor of shape (..., N, H, C).

        Returns:
            torch.Tensor: The trunked and reconcated tensor of shape (..., N, H, C).
        """
        x1, x2 = x.chunk(2, dim=-1)
        x = torch.cat((-x2, x1), dim=-1)
        return x

    def apply_rotary_pos_emb(
        self,
        x: jaxtyping.Array[torch.Tensor, "... N H C"],
        sin: jaxtyping.Array[torch.Tensor, "... N H C"],
        cos: jaxtyping.Array[torch.Tensor, "... N H C"],
        scale: jaxtyping.Array[torch.Tensor, "... N"] | None = None,
        bias: jaxtyping.Array[torch.Tensor, "... N H C"] | None = None,
    ) -> jaxtyping.Array[torch.Tensor, "... N H C"]:
        """Apply the RoPE to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (..., N, H, C).
            sin (torch.Tensor): The sine of the RoPE of shape (..., N, H, C).
            cos (torch.Tensor): The cosine of the RoPE of shape (..., N, H, C).
            scale (torch.Tensor, optional): The scaling tensor of shape (..., N).
            bias (torch.Tensor, optional): The bias tensor of shape (..., N, H, C).

        Returns:
            torch.Tensor: The rotated input tensor of shape (..., N, H, C).
        """
        if scale is None and bias is None:
            return (x * cos) + (self.trunk_half(x) * sin)
        elif scale is not None and bias is None:
            scale = einops.repeat(scale, "... N -> ... N 1 1")
            return ((x * cos) + (self.trunk_half(x) * sin)) * scale
        elif scale is None and bias is not None:
            return (x * cos) + (self.trunk_half(x) * sin) + bias
        else:
            scale = einops.repeat(scale, "... N -> ... N 1 1")
            return ((x * cos) + (self.trunk_half(x) * sin)) * scale + bias

    def forward(
        self,
        input_x: jaxtyping.Array[torch.Tensor, "... N H C"],
        positions: jaxtyping.Array[torch.Tensor, "... N D"],
        scale: jaxtyping.Array[torch.Tensor, "... N"] | None = None,
        bias: jaxtyping.Array[torch.Tensor, "... N H C"] | None = None,
    ) -> jaxtyping.Array[torch.Tensor, "... N H C"]:
        """Rotate query or key tensor by its positions. We fuse the scaling
        and adding bias into the rotation.

        Args:
            input_x (torch.Tensor): The input tensor of shape (..., N, H, C).
            positions (torch.Tensor): The positions tensor of shape (..., N, D).
            scale (torch.Tensor, optional): The scaling tensor of shape (..., N).
            bias (torch.Tensor, optional): The bias tensor of shape (..., N, H, C).

        Returns:
            torch.Tensor: The rotated input tensor of shape (..., N, H, C).
        """
        if positions in self._rope_cache:
            sin, cos = self._rope_cache[positions]
        else:
            sin, cos = self.get_sin_cos(positions)
            if len(self._rope_cache) >= self.repo_chache_maximum:
                self._rope_cache.popitem()
            self._rope_cache[positions] = (sin, cos)

        # Apply rotation
        return self.apply_rotary_pos_emb(input_x, sin, cos, scale, bias)

    def make_graph(
        self,
        input_x: jaxtyping.Array[torch.Tensor, "... N H C"],
    ) -> Digraph:
        """Make a graph of the RoPE layer.

        Args:
            input_x (torch.Tensor): The peudo input tensor of shape (..., N, H, C).

        Returns:
            Digraph: The graph of the RoPE layer.
        """
        rand_x = torch.randn_like(input_x)
        positions = torch.arange(input_x.shape[-3])[..., None]
        y = self.forward(rand_x, positions)
        return torchviz.make_dot(y.mean(), params=dict(self.named_parameters()))
