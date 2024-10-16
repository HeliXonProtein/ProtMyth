# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""This module implements the Transformer layer block, which includes
self-attention and feed-forward networks, designed for use in
sequence encoding tasks. The layer supports optional features such as
Rotary Position Encoding (RoPE) and various configurations for
normalization and gating mechanisms.

The TransformerLayer is a crucial component for building transformer
models and can be used in tasks such as natural language processing,
protein sequence modeling, and more.

Classes
-------
TransformerLayer : A class that defines a transformer layer with
                   self-attention, feed-forward networks, and
                   configurable parameters.

Dependencies
------------
- torch: For tensor operations and neural network components.
- jaxtyping: For type hinting with JAX-compatible types.
- protmyth.modules.base: Provides the BaseModule class for ProtMyth.
- protmyth.modules.register: For registering modules in the ProtMyth framework.
- protmyth.modules.seqencoders.esm.bertnorm: For ESM1 and ESM1b layer normalization.
- protmyth.modules.common.attentions: For attention mechanisms.

Parameters
----------
embed_dim : int
    The size of the input and output embeddings.
ffn_embed_dim : int
    The size of the hidden layer in the feed-forward network (FFN).
attention_heads : int
    Number of attention heads in the multi-head attention mechanism.
add_bias_kv : bool, optional
    Whether to add bias to the key and value tensors. Default is True.
use_esm1b_layer_norm : bool, optional
    Whether to use ESM1b-style layer normalization. Default is False.
use_rotary_embeddings : bool, optional
    Whether to use Rotary Position Embedding (RoPE). Default is False.
gating : bool, optional
    Whether to use gating in the attention mechanism. Default is True.
"""

from typing import Optional
import torch
from torch import nn
from jaxtyping import Float, Bool
from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module
from protmyth.modules.seqencoders.esm.bertnorm import ESM1LayerNorm, ESM1bLayerNorm
from protmyth.modules.common.attentions import Attention


@register_module("seqencoders")
class TransformerLayer(BaseModule[Float[torch.Tensor, "..."]]):
    """Transformer layer block implementing self-attention and feed-forward networks,
    with optional Rotary Position Encoding (RoPE).

    Parameters
    ----------
    embed_dim : int
        The size of the input and output embeddings.
    ffn_embed_dim : int
        The size of the hidden layer in the feed-forward network (FFN).
    attention_heads : int
        Number of attention heads in the multi-head attention mechanism.
    add_bias_kv : bool, optional
        Whether to add bias to the key and value tensors. Default is True.
    use_esm1b_layer_norm : bool, optional
        Whether to use ESM1b-style layer normalization. Default is False.
    use_rotary_embeddings : bool, optional
        Whether to use Rotary Position Embedding (RoPE). Default is False.
    gating : bool, optional
        Whether to use gating in the attention mechanism. Default is True.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        add_bias_kv: bool = True,
        use_esm1b_layer_norm: bool = False,
        use_rotary_embeddings: bool = False,
        gating: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads

        # Initialize Attention module
        self.attention = Attention(
            q_dim=embed_dim,
            kv_dim=embed_dim,
            c=embed_dim // attention_heads,
            n_head=attention_heads,
            out_dim=embed_dim,
            use_bias=add_bias_kv,
            gating=gating,
            use_rotary_embeddings=use_rotary_embeddings,
        )

        # Layer normalization
        self._init_submodules(use_esm1b_layer_norm)

    def _init_submodules(self, use_esm1b_layer_norm: bool) -> None:
        """Initialize submodules including layer normalization and FFN.


        Parameters
         ----------
        use_esm1b_layer_norm : bool
            Whether to use ESM1b-style layer normalization (True) or ESM1-style (False).

        Attributes
        ----------
        self.self_attn_layer_norm : BertLayerNorm
            Layer normalization applied before the attention mechanism.
        self.fc1 : nn.Linear
            First fully connected layer in the feed-forward network.
        self.fc2 : nn.Linear
            Second fully connected layer in the feed-forward network.
        self.final_layer_norm : BertLayerNorm
            Layer normalization applied after the feed-forward network.
        """
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(
            self,
            x: Float[torch.Tensor, "batch seq_len embed_dim"],
            self_attn_mask: Optional[Bool[torch.Tensor, "batch seq_len seq_len"]] = None,
            self_attn_padding_mask: Optional[Bool[torch.Tensor, "batch seq_len"]] = None,
    ) -> Float[torch.Tensor, "batch seq_len embed_dim"]:
        """Perform a forward pass through the Transformer layer, optionally using Rotary Position Encoding (RoPE).

        Parameters
        ----------
        x : Float[torch.Tensor, "batch seq_len embed_dim"]
            The input tensor with shape (batch, seq_len, embed_dim), representing the batch of sequences.
        self_attn_mask : Optional[Bool[torch.Tensor, "batch seq_len seq_len"]], optional
            An optional attention mask tensor with shape (batch, seq_len, seq_len).
            This mask is used to prevent attention to certain positions. Default is None.
        self_attn_padding_mask : Optional[Bool[torch.Tensor, "batch seq_len"]], optional
            An optional padding mask tensor with shape (batch, seq_len).
            This mask indicates which positions are padding and should be ignored in attention. Default is None.

        Returns
        -------
        Float[torch.Tensor, "batch seq_len embed_dim"]
            The output tensor with shape (batch, seq_len, embed_dim),
            representing the transformed sequences after passing through the layer.
        """
        residual = x
        x = self.self_attn_layer_norm(x)

        if self_attn_mask is not None and self_attn_padding_mask is not None:
            attn_mask = self_attn_mask * self_attn_padding_mask.unsqueeze(-2)
        elif self_attn_mask is not None:
            attn_mask = self_attn_mask
        elif self_attn_padding_mask is not None:
            attn_mask = self_attn_padding_mask.unsqueeze(-2)
        else:
            attn_mask = None

        x = self.attention(q_data=x, kv_data=x, attn_mask=attn_mask)

        # Add residual connection
        x = residual + x

        # Feed-forward network with residual
        residual = x
        x = self.final_layer_norm(x)
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x
