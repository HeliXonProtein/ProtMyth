# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""This module contains row attention mechanisms used in ProtMyth. Based on the Attention class from common
"""

import torch
import torchviz
from graphviz import Digraph
from torch import nn
from typing import Optional
from jaxtyping import Bool, Float
from collections.abc import Sequence
from protmyth.modules.common.attentions import Attention
from protmyth.modules.register import register_module


class RowSelfAttention(Attention):
    """Row-based self-attention mechanism inheriting from the parent Attention class."""

    def __init__(self, q_dim: int, c: int, n_head: int, out_dim: int,
                 max_tokens_per_msa: int = 1024, dropout: float = 0.1,
                 use_bias: bool = False, gating: bool = True) -> None:
        """Initializes the RowSelfAttention module, reusing most of the logic from the parent class.

        Args:
            q_dim: Size of input features.
            c: Size of channels per head.
            n_head: Number of heads.
            out_dim: Size of output features.
            max_tokens_per_msa: Threshold to batch input if size exceeds this.
            dropout: Dropout rate for attention probabilities.
            use_bias: Whether to apply bias to qkv linear.
            gating: Whether to apply a sigmoid gating for output.
        """
        super().__init__(q_dim=q_dim, kv_dim=q_dim, c=c, n_head=n_head,
                         out_dim=out_dim, use_bias=use_bias, gating=gating)
        self.max_tokens_per_msa = max_tokens_per_msa
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Float[torch.Tensor, "... Q q_dim"],
                self_attn_mask: Optional[Bool[torch.Tensor, "... #Q #Q"]] = None,
                self_attn_padding_mask: Optional[Bool[torch.Tensor, "... #Q #Q"]] = None
                ) -> Float[torch.Tensor, "... Q out_dim"]:
        """Forward pass for row-based self-attention, reusing the parent's attention mechanism.

        Args:
            x: Input tensor (..., Q, q_dim).
            self_attn_mask: Optional mask to restrict attention between certain rows.
            self_attn_padding_mask: Optional padding mask for input.

        Returns:
            Output tensor (..., Q, out_dim).
        """
        num_rows, num_cols = x.shape[-2], x.shape[-1]
        if num_rows * num_cols > self.max_tokens_per_msa and not torch.is_grad_enabled():
            return self._batched_forward(x, self_attn_mask, self_attn_padding_mask)

        return super().forward(q_data=x, kv_data=x, attn_mask=self_attn_mask)

    def _batched_forward(self, x: Float[torch.Tensor, "... Q q_dim"],
                         self_attn_mask: Optional[Bool[torch.Tensor, "... #Q #Q"]] = None,
                         self_attn_padding_mask: Optional[Bool[torch.Tensor, "... #Q #Q"]] = None,
                         ) -> Float[torch.Tensor, "... Q out_dim"]:
        """Handles large inputs by splitting them into batches.

        Args:
            x: Input tensor (..., Q, q_dim).
            self_attn_mask: Optional mask for attention.
            self_attn_padding_mask: Optional padding mask.

        Returns:
            Concatenated output from all batches.
        """
        row_chunks = torch.split(x, split_size_or_sections=self.max_tokens_per_msa // x.size(-1), dim=-2)
        output_chunks = []

        for chunk in row_chunks:
            output_chunk = super().forward(q_data=chunk, kv_data=chunk, attn_mask=self_attn_mask)
            output_chunks.append(output_chunk)

        return torch.cat(output_chunks, dim=-2)

    def make_graph(self, batch_dims: Sequence[int], q_len: int, kv_len: int, device: torch.device) -> Digraph:
        """Make a graph of the RowSelfAttention module.

        Args:
            batch_dims (Sequence[int]): The batch dimensions.
            q_len (int): Length of the query data.
            kv_len (int): Length of the key-value data.
            device (torch.device): The device for the tensors.

        Returns:
            Digraph: A Graphviz Digraph object.
        """
        q_data = torch.randn(list(batch_dims) + [q_len, self.q_dim], device=device)
        attn_mask = torch.randint(0, 2, list(batch_dims) + [q_len, q_len], device=device).bool()
        output = self.forward(q_data, self_attn_mask=attn_mask)

        return torchviz.make_dot(output.mean(), params=dict(self.named_parameters()))


@register_module("seqencoders")
class ColumnSelfAttention(Attention):
    """Compute self-attention over columns of a 2D input."""

    def __init__(
        self,
        q_dim: int,
        c: int,
        n_head: int,
        out_dim: int,
        max_tokens_per_msa: int = 1024,
        dropout: float = 0.1,
        use_bias: bool = False,
        gating: bool = True,
    ) -> None:
        """Initializes the ColumnSelfAttention module.

        Args:
            q_dim: Size of input features.
            c: Size of channels per head.
            n_head: Number of heads.
            out_dim: Size of output features.
            max_tokens_per_msa: Threshold to batch input if size exceeds this.
            dropout: Dropout rate for attention probabilities.
            use_bias: Whether to apply bias to qkv linear.
            gating: Whether to apply a sigmoid gating for output.
        """
        super().__init__(
            q_dim=q_dim,
            kv_dim=q_dim,
            c=c,
            n_head=n_head,
            out_dim=out_dim,
            use_bias=use_bias,
            gating=gating
        )
        self.max_tokens_per_msa = max_tokens_per_msa
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float[torch.Tensor, "... C q_dim"],
        self_attn_mask: Optional[Bool[torch.Tensor, "... #C #C"]] = None,
        self_attn_padding_mask: Optional[Bool[torch.Tensor, "... #C #C"]] = None,
    ) -> Float[torch.Tensor, "... C out_dim"]:
        """Forward pass for column-based self-attention.

        Args:
            x: Input tensor (..., C, q_dim).
            self_attn_mask: Optional mask to restrict attention between certain columns.
            self_attn_padding_mask: Optional padding mask for input.

        Returns:
            Output tensor (..., C, out_dim) and attention weights (optional for visualization).
        """
        num_cols = x.shape[-2]

        if num_cols > self.max_tokens_per_msa and not torch.is_grad_enabled():
            return self._batched_forward(x, self_attn_mask, self_attn_padding_mask)

        output = super().forward(q_data=x, kv_data=x, attn_mask=self_attn_mask)
        return output

    def _batched_forward(
        self,
        x: Float[torch.Tensor, "... C q_dim"],
        self_attn_mask: Optional[Bool[torch.Tensor, "... #C #C"]] = None,
        self_attn_padding_mask: Optional[Bool[torch.Tensor, "... #C #C"]] = None,
    ) -> Float[torch.Tensor, "... C out_dim"]:
        """Handles large inputs by splitting them into batches and reusing the parent's forward function.

        Args:
            x: Input tensor (..., C, q_dim).
            self_attn_mask: Optional mask for attention.
            self_attn_padding_mask: Optional padding mask.

        Returns:
            Concatenated output from all batches.
        """
        column_chunks = torch.split(x, split_size_or_sections=self.max_tokens_per_msa, dim=-2)

        output_chunks = []
        for chunk in column_chunks:
            output_chunk = super().forward(q_data=chunk, kv_data=chunk, attn_mask=self_attn_mask)
            output_chunks.append(output_chunk)

        return torch.cat(output_chunks, dim=-2)

    def make_graph(
        self,
        batch_dims: Sequence[int],
        c_len: int,
        kv_len: int,
        device: torch.device,
    ) -> Digraph:
        """Make a graph of the ColumnSelfAttention module.

        Args:
            batch_dims: Batch dimensions, corresponding to `...` in forward.
            c_len: The length of x, same as `C` in forward.
            device: The device where the tensors are located (e.g., 'cpu' or 'cuda').

        Returns:
            A Graphviz Digraph object representing the attention computation graph.
        """
        x_data = torch.randn(list(batch_dims) + [c_len, self.q_dim], device=device)
        attn_mask = torch.randint(0, 2, list(batch_dims) + [c_len, c_len], device=device).bool()
        output = self.forward(x_data, self_attn_mask=attn_mask)

        return torchviz.make_dot(output.mean(), params=dict(self.named_parameters()))
