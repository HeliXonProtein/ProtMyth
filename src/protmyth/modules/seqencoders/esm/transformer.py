# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

from protmyth.modules.base import BaseModule

import torch
import torch.nn as nn
import einops

from typing import Tuple, Union, Any
from jaxtyping import Float, Array, Bool


class NormalizedResidualBlock(BaseModule):
    """Normalized Residual Block.

    This class implements a residual block with layer normalization and dropout.

    Parameters
    ----------
    layer : nn.Module
        The sub-module to be wrapped in the residual block.
    embedding_dim : int
        The dimensionality of the input embeddings.
    dropout : float, optional
        Dropout probability applied to the output of the layer, by default 0.1.
    """

    def __init__(
        self,
        layer: nn.Module,
        embedding_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layer = layer
        self.dropout_module = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len embed_dim"],
        *args: Any,
        **kwargs: Any,
    ) -> Union[Float[torch.Tensor, "batch seq_len embed_dim"],
               Tuple[Float[torch.Tensor, "batch seq_len embed_dim"], ...]]:
        """Forward pass through the Normalized Residual Block.

        Parameters
        ----------
        x : Float[Array, "batch seq_len embed_dim"]
            Input tensor.
        *args : Any
            Additional positional arguments for the layer.
        **kwargs : Any
            Additional keyword arguments for the layer.

        Returns
        -------
        Union[Float[Array, "batch seq_len embed_dim"], Tuple[Float[Array, "batch seq_len embed_dim"], ...]]
            Output tensor, possibly with additional outputs from the layer.
        """
        residual = x
        x = self.layer_norm(x)
        outputs = self.layer(x, *args, **kwargs)

        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x = outputs
            out = None

        x = self.dropout_module(x)
        x = residual + x

        return (x,) + tuple(out) if out is not None else x


class TransformerLayer(BaseModule):
    """Transformer layer block.

    This class implements a single transformer layer, including self-attention
    and a feed-forward network, with residual connections and layer normalization.

    Parameters
    ----------
    embed_dim : int
        The dimensionality of the input embeddings.
    ffn_embed_dim : int
        The dimensionality of the feed-forward network's hidden layer.
    attention_heads : int
        Number of attention heads for the self-attention mechanism.
    add_bias_kv : bool, optional
        Whether to add bias to key and value projections, by default True.
    use_esm1b_layer_norm : bool, optional
        Whether to use ESM1b layer normalization, by default False.
    use_rotary_embeddings : bool, optional
        Whether to use rotary embeddings, by default False.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        add_bias_kv: bool = True,
        use_rotary_embeddings: bool = False,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self.dropout_prob = 0.1  # Assume some dropout value

        # Initialize self-attention
        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )

        # Initialize layer norms
        self.self_attn_layer_norm = nn.Layernorm(self.embed_dim)
        self.final_layer_norm = nn.Layernorm(self.embed_dim)

        # Initialize feedforward layers
        fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        self.feed_forward_layer = NormalizedResidualBlock(
            nn.Sequential(fc1, nn.GELU(), fc2),
            self.embed_dim,
            self.dropout_prob,
        )


    def forward(
        self,
        x: Float[Array, "batch seq_len embed_dim"],
        self_attn_mask: Union[Float[Array, "batch seq_len seq_len"], None] = None,
        self_attn_padding_mask: Union[Float[Array, "batch seq_len"], None] = None,
        need_head_weights: bool = False
    ) -> Union[Float[Array, "batch seq_len embed_dim"],
               Tuple[Float[Array, "batch seq_len embed_dim"],
                     Float[Array, "batch heads seq_len seq_len"]]]:
        """Forward pass through the Transformer layer.

        Parameters
        ----------
        x : Float[Array, "batch seq_len embed_dim"]
            Input tensor.
        self_attn_mask : Union[Float[Array, "batch seq_len seq_len"], None], optional
            Mask for self-attention, by default None.
        self_attn_padding_mask : Union[Float[Array, "batch seq_len"], None], optional
            Padding mask for self-attention, by default None.
        need_head_weights : bool, optional
            Whether to return attention weights, by default False.

        Returns
        -------
        Union[Float[Array, "batch seq_len embed_dim"],
              Tuple[Float[Array, "batch seq_len embed_dim"],
              Float[Array, "batch heads seq_len seq_len"]]]
        Output tensor and optionally attention weights.
        """
        # Self-attention with residual and layer norm
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        # Feedforward network with residual and layer norm
        x = self.feed_forward_layer(x)

        return (x, attn) if need_head_weights else x