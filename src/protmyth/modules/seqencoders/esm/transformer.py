# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

from typing import Optional, Tuple
import torch
from torch import nn
import einops
from jaxtyping import Float, Bool
from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module
from protmyth.modules.seqencoders.esm.bertnorm import ESM1LayerNorm, ESM1bLayerNorm
from protmyth.modules.common.attentions import Attention
from protmyth.modules.common.rotary_embedding import RotaryEmbedding  # Import the new RoPE class


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
        self.use_rotary_embeddings = use_rotary_embeddings

        # Initialize Attention module
        self.attention = Attention(
            q_dim=embed_dim,
            kv_dim=embed_dim,
            c=embed_dim // attention_heads,
            n_head=attention_heads,
            out_dim=embed_dim,
            use_bias=add_bias_kv,
            gating=gating
        )

        # Layer normalization
        self._init_submodules(use_esm1b_layer_norm)

        # Rotary Positional Encoding (RoPE) initialization
        if self.use_rotary_embeddings:
            self.rope = RotaryEmbedding(self.embed_dim // self.attention_heads)

    def _init_submodules(self, use_esm1b_layer_norm: bool) -> None:
        """Initialize submodules including layer normalization and FFN."""
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len embed_dim"],
        positions: Float[torch.Tensor, "batch seq_len position_dim"],
        self_attn_mask: Optional[Bool[torch.Tensor, "batch seq_len seq_len"]] = None,
        self_attn_padding_mask: Optional[Bool[torch.Tensor, "batch seq_len"]] = None,
    ) -> Float[torch.Tensor, "batch seq_len embed_dim"]:
        """Forward pass through the Transformer layer with optional RoPE.
        Parameters
        ----------
        **kwargs : Any
            Keyword arguments. Accepts:
                x : torch.Tensor
                    Input tensor of shape (batch, seq_len, embed_dim).
                self_attn_mask : Optional[torch.Tensor], optional
                    Attention mask tensor of shape (batch, seq_len, seq_len). Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        residual = x
        x = self.self_attn_layer_norm(x)

        if self.use_rotary_embeddings:
            # Apply RoPE to queries and keys using the updated RoPE mechanism
            q, k = self._apply_rope_to_qk(x, x)
            x = self.attention(q_data=q, kv_data=k, attn_mask=self_attn_mask)
        else:
            x = self.attention(q_data=x, kv_data=x, attn_mask=self_attn_mask)

        # Add residual connection
        x = residual + x

        # Feed-forward network with residual
        residual = x
        x = self.final_layer_norm(x)
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x

    def _apply_rope_to_qk(
        self,
        q: Float[torch.Tensor, "batch seq_len embed_dim"],
        k: Float[torch.Tensor, "batch seq_len embed_dim"]
    ) -> Tuple[Float[torch.Tensor, "batch seq_len embed_dim"], Float[torch.Tensor, "batch seq_len embed_dim"]]:
        """Apply RoPE (Rotary Position Embedding) to the query and key tensors using the new RotaryEmbedding.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor of shape (batch, seq_len, embed_dim).
        k : torch.Tensor
            Key tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        q_rot : torch.Tensor
            Rotated query tensor of shape (batch, seq_len, embed_dim).
        k_rot : torch.Tensor
            Rotated key tensor of shape (batch, seq_len, embed_dim).
        """
        # Reshape to [..., sequence_length, n_head, c]
        q_reshaped = einops.rearrange(q, 'b n (h d) -> b n h d', h=self.attention_heads)
        k_reshaped = einops.rearrange(k, 'b n (h d) -> b n h d', h=self.attention_heads)

        # Apply RoPE to queries and keys
        q_rope, k_rope = self.rope(q_reshaped, k_reshaped)

        # Rearrange back to original shape (batch, seq_len, embed_dim)
        q_rot = einops.rearrange(q_rope, 'b n h d -> b n (h d)')
        k_rot = einops.rearrange(k_rope, 'b n h d -> b n (h d)')

        return q_rot, k_rot
