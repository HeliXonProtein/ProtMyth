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

from typing import Optional, Tuple
from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module


@register_module("common")
class Attention(BaseModule[Float[torch.Tensor, "..."]]):
    """Common attention module, now it supports:
        1.Multi-head
        2.qk scaling
        3.attention mask
    """

    def __init__(
        self,
        q_dim: int = 32,
        kv_dim: int = 32,
        c: int = 8,
        n_head: int = 4,
        out_dim: int = 32,
        use_bias: bool = False,
        gating: bool = True,
        use_rotary_embeddings: bool = False,
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
            use_rotary_embeddings: Whether to use RoPE as the positional embedding.
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

        self.rot_emb = None
        if use_rotary_embeddings:
            self.rot_emb = RotaryEmbedding(dim=self.c)

    def _qk_scale(
        self,
    ) -> Float[torch.Tensor, "..."]:
        """scaling for attention logits
        """
        return self.c ** 0.5

    def forward(
        self,
        q_data: Float[torch.Tensor, "... Q q_dim"],
        kv_data: Float[torch.Tensor, "... K kv_dim"],
        attn_mask: Optional[Bool[torch.Tensor, "... #Q #K"]] = None,
    ) -> Float[torch.Tensor, "... Q out_dim"]:
        """Forward process of multi-head attention

        Args:
            q_data: Query features, (..., Q, q_dim).
            kv_data: Key and value features, (..., K, kv_dim).
            attn_mask: Attention mask with size (..., Q/1, K/1), or None
        Returns:
            Output features, (..., Q, out_dim).
        """
        q = einops.rearrange(self.q_linear(q_data), '... Q (H C) -> ... Q H C', C=self.c)
        k, v = einops.rearrange(self.kv_linear(kv_data), '... K (split H C) -> split ... K H C', split=2, C=self.c)

        if self.rot_emb:
            q, k = self.rot_emb(q, k)

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


@register_module("common")
class RotaryEmbedding(nn.Module):
    """Implements rotary positional embedding for transformer models."""

    def __init__(self, dim: int, *_, **__) -> None:
        """Initialize RotaryEmbedding Layer.

        Args:
            dim: The dimensionality of the embedding.
        """
        super().__init__()

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._cos_cached = torch.zeros(1)
        self._sin_cached = torch.zeros(1)

    def rotate_half(
        self,
        x: Float[torch.Tensor, '... c'],
    ) -> Float[torch.Tensor, '... c']:
        """Rotate half of the tensor along the last dimension.

        Args:
            x: Input tensor of shape [..., c].

        Returns:
            Tensor after rotating half of its components.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self, x: Float[torch.Tensor, '... s c'],
        cos: Float[torch.Tensor, 's c'],
        sin: Float[torch.Tensor, 's c'],
    ) -> Float[torch.Tensor, '... s c']:
        """Apply rotary positional embeddings to the input tensor.

        Args:
            x: Input tensor of shape [..., sequence_length, embed_dim].
            cos: Cosine values for the rotary embeddings.
            sin: Sine values for the rotary embeddings.

        Returns:
            Tensor with applied rotary positional embeddings.
        """
        cos = cos[:x.shape[-2], :]
        sin = sin[:x.shape[-2], :]
        return (x * cos) + (self.rotate_half(x) * sin)

    def _update_cos_sin_tables(
        self,
        x: Float[torch.Tensor, '... c'],
        seq_dimension: int = -2
    ) -> Tuple[Float[torch.Tensor, 's c'], Float[torch.Tensor, 's c']]:
        """Update cached cosine and sine values based on the input tensor sequence length.

        Args:
            x: Input tensor.
            seq_dimension: The dimension along which the sequence length is determined.

        Returns:
            Cached cosine and sine tensors.
        """
        seq_len = x.shape[seq_dimension]

        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()

        return self._cos_cached, self._sin_cached

    def forward(
        self,
        q: Float[torch.Tensor, '... s n c'],
        k: Float[torch.Tensor, '... s n c'],
    ) -> Tuple[Float[torch.Tensor, '... s n c'], Float[torch.Tensor, '... s n c']]:
        """Forward pass for the RotaryEmbedding.

        Args:
            q: Input tensor of size [..., sequence_length, n_head, c].
            k: Input tensor of size [..., sequence_length, n_head, c].

        Returns:
            Output tensors (q_rope, k_rope) with sinusoidal positional embeddings,
            having the same shape as q and k.
        """
        # Transpose to shape [..., n_head, sequence_length, c]
        q = einops.rearrange(q, '... s n c -> ... n s c')
        k = einops.rearrange(k, '... s n c -> ... n s c')

        # Update cosine and sine tables
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        # Apply rotary positional embeddings
        q_rope = self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)
        k_rope = self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)

        # Transpose back to original shape
        return (
            einops.rearrange(q_rope, '... n s c -> ... s n c'),
            einops.rearrange(k_rope, '... n s c -> ... s n c'),
        )
