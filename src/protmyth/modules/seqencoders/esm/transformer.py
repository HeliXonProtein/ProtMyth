import torch
import torch.nn as nn
import einops
from jaxtyping import Float, Bool
from typing import Optional, Tuple
from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module
from protmyth.modules.seqencoder.bertnorm import ESM1LayerNorm,ESM1bLayerNorm 

@register_module("seqencoders")
class TransformerLayer(BaseModule[Float[torch.Tensor, "..."]]):
    """Transformer layer block implementing self-attention and feed-forward networks.

    Parameters
    ----------
    embed_dim : int
        Dimension of embedding space.
    ffn_embed_dim : int
        Dimension of feed-forward network's hidden layer.
    attention_heads : int
        Number of attention heads.
    add_bias_kv : bool, optional
        Whether to apply bias to key and value linear layers, by default True.
    use_esm1b_layer_norm : bool, optional
        Whether to use ESM1b style layer normalization, by default False.
    use_rotary_embeddings : bool, optional
        Whether to use rotary embeddings, by default False.
    gating : bool, optional
        Whether to apply gating mechanism on output, by default True.
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

        # Initialize Attention module parameters
        self.q_dim = embed_dim
        self.kv_dim = embed_dim
        self.c = embed_dim // attention_heads
        self.n_head = attention_heads
        self.out_dim = embed_dim
        self.gating = gating

        self._init_submodules(use_esm1b_layer_norm, add_bias_kv)

    def _init_submodules(self, use_esm1b_layer_norm: bool, add_bias_kv: bool) -> None:
        """Initialize submodules including attention, layer normalization, and feed-forward network.

        Parameters
        ----------
        use_esm1b_layer_norm : bool
            Whether to use ESM1b style layer normalization.
        add_bias_kv : bool
            Whether to add bias to key-value linear projections.
        """
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        # Self-attention using Attention module
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        # Feed-forward network
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len embed_dim"],
        self_attn_mask: Optional[Bool[torch.Tensor, "batch seq_len seq_len"]] = None,
        self_attn_padding_mask: Optional[Bool[torch.Tensor, "batch seq_len"]] = None,
        need_head_weights: bool = False
    ) -> Tuple[Float[torch.Tensor, "batch seq_len embed_dim"], Optional[Float[torch.Tensor, "batch heads seq_len seq_len"]]]:
        """Forward pass through the transformer layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, seq_len, embed_dim).
        self_attn_mask : Optional[torch.Tensor], optional
            Mask for self-attention mechanism with shape (batch, seq_len, seq_len), by default None.
        self_attn_padding_mask : Optional[torch.Tensor], optional
            Padding mask for self-attention with shape (batch, seq_len), by default None.
        need_head_weights : bool, optional
            Whether to return attention weights for all heads, by default False.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Output tensor of shape (batch, seq_len, embed_dim) and attention weights if requested.
        """
        # Self-attention mechanism
        residual = x
        x = self.self_attn_layer_norm(x)

        # Self-attention using multi-head attention
        q = einops.rearrange(self.q_linear(x), 'b seq (h c) -> b seq h c', h=self.n_head)
        k, v = einops.rearrange(self.kv_linear(x), 'b seq (split h c) -> split b seq h c', split=2, h=self.n_head)
        logits = torch.einsum('bqhc,bkhc->bhqk', q, k) / self.c**0.5

        if self_attn_mask is not None:
            logits = torch.where(self_attn_mask.unsqueeze(1), logits, float('-inf'))

        attn_weights = torch.softmax(logits, dim=-1)
        weighted_avg = torch.einsum('bhqk,bkhc->bqhc', attn_weights, v)
        weighted_avg = einops.rearrange(weighted_avg, 'b q h c -> b q (h c)')

        if self.gating:
            gate_values = torch.sigmoid(self.gating_linear(x))
            weighted_avg *= gate_values

        output = self.output_linear(weighted_avg)
        x = residual + output

        # Feed-forward network
        residual = x
        x = self.final_layer_norm(x)
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn_weights if need_head_weights else None

