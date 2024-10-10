# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

from typing import Tuple
import torch
import torch.nn as nn

from protmyth.modules.register import register_module


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[: x.shape[-2], :]
    sin = sin[: x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)

@register_module("common")
class RotaryEmbedding(nn.Module):
    """
    todo
    """

    def __init__(self, dim: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=-2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

        return self._cos_cached, self._sin_cached


    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the RoPEEmbedding.

        Parameters
        ----------
        q : torch.Tensor
            Input tensor of size [..., sequence_length, n_head, c].
        k : torch.Tensor
            Input tensor of size [..., sequence_length, n_head, c].

        Returns
        -------
        (q_rope, k_rope): Tuple
            Output tensor with sinusoidal positional embeddings.
            q_rope & k_rope has the same shape as q & k.
        """

        q = q.transpose(-3, -2) # [..., sequence_length, n_head, c] -> [..., n_head, sequence_length, c]
        k = k.transpose(-3, -2) # [..., sequence_length, n_head, c] -> [..., n_head, sequence_length, c]

        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        q_rope = apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)
        k_rope = apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)

        return (
            q_rope.transpose(-3, -2),
            k_rope.transpose(-3, -2),
        )
