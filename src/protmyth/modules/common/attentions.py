# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""This module contains attention mechanisms used in ProtMyth. Based on the attention mechanism used,
we divide the attention mechanism into three categories: (https://arxiv.org/pdf/2203.14263)

1. Feature-Related:
    a. Multiplicity.
    b. Levels.
    c. Representations.
2. General:
    a. Scoring.
    b. Alignment.
    c. Dimensionality.
3. Query-Related:
    a. Type.
    b. Multiplicity.

We will implement the following attention mechanisms and transform them into protein based modules:

1. Place holder.
"""

import einops
import torch
from torch import nn
from jaxtyping import Float, Bool
import torchviz
from graphviz import Digraph

from typing import Optional
from collections.abc import Sequence
from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module


import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple
from jaxtyping import Float, Array


@register_module("common")
class RotaryEmbedding(nn.Module):
    """
    ROPE Rotary position embeddings from RoFormer.

    This method rotates query and key tensors using rotation matrices that
    depend on the relative positions, which helps capture positional information
    in a manner conducive to transformer architectures.

    References
    ----------
    - RoFormer: https://arxiv.org/abs/2104.09864
    - GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    - RoFormer repo: https://github.com/ZhuiyiTechnology/roformer

    Notes
    -----
    This embedding layer is not registered as a traditional model layer
    because it does not create new embedding dimensions but modifies the
    positional aspects of given embeddings.

    Parameters
    ----------
    dim : int
        The dimension size for embeddings.

    Attributes
    ----------
    inv_freq : torch.Tensor
        Inverse frequency for rotational embeddings.
    _seq_len_cached : Optional[int]
        Cached sequence length to minimize recomputation.
    _cos_cached : Optional[torch.Tensor]
        Cached cosine embedding calculations.
    _sin_cached : Optional[torch.Tensor]
        Cached sine embedding calculations.
    """

    def __init__(self, dim: int, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None


    def _rotate_half(self, x: Float[Array, "batch seq_len dim"]) -> Float[Array, "batch seq_len dim"]:
        """Rotate the halves of the input tensor.

        This function splits the input tensor into two halves along the last dimension,
        then swaps and negates the second half before concatenating them back together.

        Parameters
        ----------
        x : Float[Array, "batch seq_len dim"]
            Input tensor to be rotated.

        Returns
        -------
        Float[Array, "batch seq_len dim"]
            Tensor with rotated halves.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(
            self,
            x: Float[Array, "batch seq_len dim"],
            cos: Float[Array, "1 seq_len half_dim"],
            sin: Float[Array, "1 seq_len half_dim"]
    ) -> Float[Array, "batch seq_len dim"]:
        """Apply rotary positional embeddings to the input tensor.

        This function applies the rotary positional embeddings by combining the input
        tensor with cosine and sine components, effectively rotating the tensor in
        the embedding space.

        Parameters
        ----------
        x : Float[Array, "batch seq_len dim"]
            Input tensor to which the rotary embeddings are applied.
        cos : Float[Array, "1 seq_len half_dim"]
            Cosine components of the rotary embeddings.
        sin : Float[Array, "1 seq_len half_dim"]
            Sine components of the rotary embeddings.

        Returns
        -------
        Float[Array, "batch seq_len dim"]
            Tensor with applied rotary positional embeddings.
        """
        cos = cos[:, :x.shape[-2], :]
        sin = sin[:, :x.shape[-2], :]

        return (x * cos) + (self.rotate_half(x) * sin)

    def _update_cos_sin_tables(self, x: Float[Array, "batch seq_len ..."], seq_dimension: int = 1) -> Tuple[
        Float[Array, "1 seq_len half_dim"], Float[Array, "1 seq_len half_dim"]]:
        """
        Update cached cosine and sine tables based on the input tensor dimensions.

        Parameters
        ----------
        x : Float[Array, "batch seq_len ..."]
            Input tensor for which to compute the positional encodings.
        seq_dimension : int, optional
            Dimension containing the sequence length information, by default 1.

        Returns
        -------
        Tuple[Float[Array, "1 seq_len half_dim"], Float[Array, "1 seq_len half_dim"]]
            Cosine and sine positional encodings.
        """
        seq_len = x.shape[seq_dimension]

        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len

            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)

            emb = rearrange(freqs, 'seq half_dim -> 1 seq half_dim')

            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

        return self._cos_cached, self._sin_cached


    def forward(
            self,
            q: Float[Array, "batch seq_len dim"],
            k: Float[Array, "batch seq_len dim"],
    ) -> Tuple[Float[Array, "batch seq_len dim"],
               Float[Array, "batch seq_len dim"]]:
        """Apply rotary positional embeddings on the input queries and keys.

        Parameters
        ----------
        q : Float[Array, "batch seq_len dim"]
            Query tensor to which the rotary embeddings are to be applied.
        k : Float[Array, "batch seq_len dim"]
            Key tensor to which the rotary embeddings are to be applied.

        Returns
        -------
        Tuple[Float[Array, "batch seq_len dim"], Float[Array, "batch seq_len dim"]]
            The rotated query and key tensors.
        """
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        q_rotated = self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)
        k_rotated = self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)

        return q_rotated, k_rotated


@register_module("common")
class Attention(BaseModule[Float[torch.Tensor, "..."]]):
    """Common attention module, now it supports:
        1.Multi-head
        2.qk scaling
        3.attention mask
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        c: int,
        n_head: int,
        out_dim: int,
        use_bias: bool = False,
        gating: bool = True,
    ) -> None:
        """Attention initialization.

        Args:
            q_dim: Size of input query features.
            kv_dim: Size of input key and value features.
            c: Size of channels per head.
            n_head: Number of heads.
            out_dim: Size of output features.
            use_bias: Whether to apply bias to qkv linear.
            gating: Whether to apply a sigmoid gating for output.
        """
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.c = c
        self.n_head = n_head
        self.out_dim = out_dim
        self.gating = gating

        self.q_linear = nn.Linear(
            in_features=q_dim,
            out_features=n_head * c,
            bias=use_bias,
        )
        self.kv_linear = nn.Linear(
            in_features=kv_dim,
            out_features=n_head * c * 2,
            bias=use_bias,
        )

        if gating:
            self.gating_linear = nn.Linear(
                in_features=q_dim,
                out_features=n_head * c,
                bias=True,
            )

        self.output_linear = nn.Linear(
            in_features=n_head * c,
            out_features=out_dim,
            bias=True,
        )

    def _qk_scale(
        self,
    ) -> Float[torch.Tensor, "..."]:
        """scaling for attention logits
        """
        return self.c ** 0.5

    def forward(
        self,
        q_data: Float[torch.Tensor, "... qry_len q_dim"],
        kv_data: Float[torch.Tensor, "... key_len kv_dim"],
        attn_mask: Optional[Bool[torch.Tensor, "... #qry_len #key_len"]] = None,
    ) -> Float[torch.Tensor, "... qry_len out_dim"]:
        """Forward process of multi-head attention

        Args:
            q_data: Query features, (..., qry_len, q_dim).
            kv_data: Key and value features, (..., key_len, kv_dim).
            attn_mask: Attention mask with size (..., qry_len/1, key_len/1), or None
        Returns:
            Output features, (..., qry_len, out_dim).
        """
        q = einops.rearrange(self.q_linear(q_data), '... Q (H C) -> ... Q H C', C=self.c)
        k, v = einops.rearrange(self.kv_linear(kv_data), '... K (split H C) -> split ... K H C', split=2, C=self.c)

        logits = torch.einsum("...qhc,...khc->...hqk", q, k) * self._qk_scale()

        if attn_mask is not None:
            logits = torch.where(attn_mask.unsqueeze(-3), logits, -1e9)

        weights = nn.functional.softmax(logits, dim=-1)
        weighted_avg = torch.einsum("...hqk,...khc->...qhc", weights, v)
        weighted_avg = einops.rearrange(weighted_avg, '... Q H C -> ... Q (H C)')

        if self.gating:
            gate_values = torch.sigmoid(self.gating_linear(q_data))
            weighted_avg = weighted_avg * gate_values

        output = self.output_linear(weighted_avg)

        return output

    def make_graph(
        self,
        batch_dims: Sequence[int],
        q_len: int,
        kv_len: int,
        device: torch.device,
    ) -> Digraph:
        """Make a graph of the attention module.

        Args:
            batch_dims: batch_dims, same as ... in forward
            q_len: the length of q_data, same as Q in forward
            kv_len: the length of kv_data, same as K in forward
            device: the device of tensor
        Returns:
            Output Digraph: the graph of the attention module with random initialization.
        """
        q_data = torch.randn(list(batch_dims) + [q_len, self.q_dim], device=device)
        kv_data = torch.randn(list(batch_dims) + [kv_len, self.kv_dim], device=device)
        attn_mask = torch.randint(0, 2, list(batch_dims) + [q_len, kv_len], device=device).bool()
        output = self.forward(q_data, kv_data, attn_mask=attn_mask)
        return torchviz.make_dot(output.mean(), params=dict(self.named_parameters()))
