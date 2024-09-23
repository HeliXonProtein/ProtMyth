# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

from .axial_attention import ColumnSelfAttention, RowSelfAttention
from protmyth.modules.base import BaseModule
import torch
import torch.nn as nn
from typing import Optional
"""This module implements components of an Axial MSA Transformer.

Classes:
    FeedForwardNetwork: Implements a feed-forward network with activation and dropout layers.
    AxialTransformerLayer: Defines an Axial Transformer block with row-wise and column-wise self-attention 
                           mechanisms, as well as a feed-forward network.

Modules:
    RowSelfAttention (imported): Handles self-attention across rows of a multi-dimensional input.
    ColumnSelfAttention (imported): Handles self-attention across columns of a multi-dimensional input.

Dependencies:
    torch: PyTorch library for tensor computations and deep learning.
    nn.Module: Used to define neural network layers and models.
    BaseModule: A base class that other models extend from.
    NormalizedResidualBlock: A helper class to build residual blocks with normalization.

Usage:
    Use AxialTransformerLayer to apply both row and column self-attention followed by a feed-forward network
    on multi-dimensional sequence inputs (e.g., MSAs). Residual connections and normalization are built into
    the architecture.
"""

class FeedForwardNetwork(BaseModule):
    """Implements a FeedForward Network with activation and dropout layers.

    This module takes in an input tensor, applies a linear transformation, 
    activation, and dropout, and then transforms it back to the original 
    embedding dimension.

    Args:
        embedding_dim (int): The dimensionality of input embeddings.
        ffn_embedding_dim (int): The dimensionality of the hidden layer in the feed-forward network.
        activation_dropout (float): Dropout probability after the activation function.
        max_tokens_per_msa (int): Maximum number of tokens allowed per MSA input.
    """
    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        activation_dropout: float = 0.1,
        max_tokens_per_msa: int = 2**14,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.max_tokens_per_msa = max_tokens_per_msa
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(activation_dropout)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the feed-forward network.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments (including input tensor `x`).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        # Extract x from args or kwargs
        x = kwargs.get('x', args[0] if args else None)

        if x is None:
            raise ValueError("Input tensor 'x' must be provided as the first positional argument or as a keyword argument.")

        # Feed-forward logic
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x



class AxialTransformerLayer(BaseModule):
    """Implements an Axial MSA Transformer block.

    This layer includes row-wise and column-wise self-attention mechanisms 
    followed by a feed-forward network. Residual connections and LayerNorm 
    are applied to each component.

    Args:
        embedding_dim (int): The dimensionality of input embeddings.
        ffn_embedding_dim (int): The dimensionality of the feed-forward network's hidden layer.
        num_attention_heads (int): Number of attention heads for self-attention layers.
        dropout (float): Dropout probability applied to the output of each layer.
        attention_dropout (float): Dropout probability within the attention mechanism.
        activation_dropout (float): Dropout probability applied after activation in the feed-forward network.
        max_tokens_per_msa (int): Maximum number of tokens allowed per MSA input.
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_tokens_per_msa: int = 2**14,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout

        # Initialize row-wise and column-wise self-attention layers
        row_self_attention = RowSelfAttention(
            embedding_dim,
            num_attention_heads,
            dropout=dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        column_self_attention = ColumnSelfAttention(
            embedding_dim,
            num_attention_heads,
            dropout=dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        # Initialize feed-forward network
        feed_forward_layer = FeedForwardNetwork(
            embedding_dim,
            ffn_embedding_dim,
            activation_dropout=activation_dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        # Wrap layers in residual connections
        self.row_self_attention = self.build_residual(row_self_attention)
        self.column_self_attention = self.build_residual(column_self_attention)
        self.feed_forward_layer = self.build_residual(feed_forward_layer)

    def build_residual(self, layer: nn.Module) -> nn.Module:
        """Wraps a layer in a NormalizedResidualBlock with LayerNorm and residual connections.

        Args:
            layer (nn.Module): The module to be wrapped in residual connections.

        Returns:
            nn.Module: A NormalizedResidualBlock wrapping the input layer.
        """
        return NormalizedResidualBlock(
            layer,
            self.embedding_dim,
            self.dropout_prob,
        )

    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the AxialTransformerLayer.

        Applies row-wise and column-wise self-attention followed by a feed-forward network.
        LayerNorm and residual connections are applied before or after each component.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments (x, self_attn_mask, self_attn_padding_mask, need_head_weights).

        Returns:
            torch.Tensor: The output tensor after passing through the layer.
            If need_head_weights is True, also returns attention weights.
        """
        # Extract the specific arguments from kwargs or provide defaults
        x = kwargs.get('x')
        self_attn_mask = kwargs.get('self_attn_mask', None)
        self_attn_padding_mask = kwargs.get('self_attn_padding_mask', None)
        need_head_weights = kwargs.get('need_head_weights', False)

        # Perform the forward pass as before
        x, row_attn = self.row_self_attention(
            x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        x, column_attn = self.column_self_attention(
            x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        x = self.feed_forward_layer(x)
        
        if need_head_weights:
            return x, column_attn, row_attn
        else:
            return x
