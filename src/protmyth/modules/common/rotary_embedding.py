# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

from typing import Tuple
import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from protmyth.modules.register import register_module


def rotate_half(x: Float[...]) -> Float[...] :
    """Rotate half of the tensor along the last dimension.
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [..., 2 * d].
    Returns
    -------
    torch.Tensor
        Tensor after rotating half of its components.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: Float[...], cos: Float[...], sin: Float[...]) -> Float[...] :
    """Apply rotary positional embeddings to the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [..., sequence_length, embed_dim].
    cos : torch.Tensor
        Cosine values for the rotary embeddings.
    sin : torch.Tensor
        Sine values for the rotary embeddings.

    Returns
    -------
    torch.Tensor
        Tensor with applied rotary positional embeddings.
    """
    cos = cos[:x.shape[-2], :]
    sin = sin[:x.shape[-2], :]
    return (x * cos) + (rotate_half(x) * sin)


@register_module("common")
class RotaryEmbedding(nn.Module):
    """Implements rotary positional embedding for transformer models.
    """

    def __init__(self, dim: int, *args, **kwargs) -> None:
        """Initialize RotaryEmbedding Layer.

        Parameters
        ----------
        dim : int
            The dimensionality of the embedding.
        """
        super().__init__()

        # Generate and save the inverse frequency buffer (non-trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x: Float[...], seq_dimension: int = -2) -> Tuple[Float[...], Float[...]]:
        """Update cached cosine and sine values based on the input tensor sequence length.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        seq_dimension : int
            The dimension along which the sequence length is determined.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Cached cosine and sine tensors.
        """
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed or if we're on a new device
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

        return self._cos_cached, self._sin_cached

    def forward(self, q: Float[...], k: Float[...]) -> Tuple[Float[...], Float[...]]:
        """Forward pass for the RotaryEmbedding.

        Parameters
        ----------
        q : torch.Tensor
            Input tensor of size [..., sequence_length, n_head, c].
        k : torch.Tensor
            Input tensor of size [..., sequence_length, n_head, c].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Output tensors (q_rope, k_rope) with sinusoidal positional embeddings,
            having the same shape as q and k.
        """
        # Transpose to shape [..., n_head, sequence_length, c]
        q = rearrange(q, '... s n c -> ... n s c')
        k = rearrange(k, '... s n c -> ... n s c')

        # Update cosine and sine tables
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        # Apply rotary positional embeddings
        q_rope = apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)
        k_rope = apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)

        # Transpose back to original shape
        return (
            rearrange(q_rope, '... n s c -> ... s n c'),
            rearrange(k_rope, '... n s c -> ... s n c'),
        )
