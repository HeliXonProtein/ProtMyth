# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

import pytest
import torch
import esm
import einops

from protmyth.modules.seqencoders.esm.transformer import TransformerLayer
from protmyth.adaptors.esm.transformer import transformer_mapper


@pytest.mark.parametrize(
    "embed_dim, ffn_embed_dim, attention_heads, add_bias_kv,"
    "use_esm1b_layer_norm, use_rotary_embeddings, gating, batch_size, seq_len",
    [
        (64, 128, 8, True, False, True, False, 2, 10),
        # (128, 256, 16, False, False, True, False, 4, 20),
        # (256, 512, 32, False, False, True, False, 8, 30),
    ]
)
def test_transformer_mapper(
    embed_dim: int,
    ffn_embed_dim: int,
    attention_heads: int,
    add_bias_kv: bool,
    use_esm1b_layer_norm: bool,
    use_rotary_embeddings: bool,
    gating: bool,
    batch_size: int,
    seq_len: int
):
    """Test the TransformerLayer forward pass with various input configurations.

    Parameters
    ----------
    embed_dim : int
        Dimension of the input and output embeddings.
    ffn_embed_dim : int
        Dimension of the hidden layer in the feed-forward network.
    attention_heads : int
        Number of attention heads in the multi-head attention mechanism.
    add_bias_kv : bool
        Whether to add bias to the key and value tensors.
    use_esm1b_layer_norm : bool
        Whether to use ESM1b-style layer normalization.
    use_rotary_embeddings : bool
        Whether to use Rotary Position Encoding (RoPE).
    gating : bool
        Whether to use gating in the attention mechanism.
    batch_size : int
        The batch size for the input tensor.
    seq_len : int
        The sequence length for the input tensor.
    """
    layer_src = esm.modules.TransformerLayer(
        embed_dim=embed_dim,
        ffn_embed_dim=ffn_embed_dim,
        attention_heads=attention_heads,
        add_bias_kv=False,
        use_esm1b_layer_norm=use_esm1b_layer_norm,
        use_rotary_embeddings=use_rotary_embeddings,
    )
    layer_tgt = TransformerLayer(
        embed_dim=embed_dim,
        ffn_embed_dim=ffn_embed_dim,
        attention_heads=attention_heads,
        add_bias_kv=add_bias_kv,
        use_esm1b_layer_norm=use_esm1b_layer_norm,
        use_rotary_embeddings=use_rotary_embeddings,
        gating=gating,
    )
    # Initialize source layer parameters
    for p in layer_src.parameters():
        p.data.normal_(0.0, 0.5)

    # Map states between the two layers
    mapper = transformer_mapper(tgt_pfx="", src_pfx="")
    result = mapper(layer_tgt.state_dict(), layer_src.state_dict())
    assert set(result.matched_source) == set(layer_src.state_dict().keys())
    assert set(result.matched_target) == set(layer_tgt.state_dict().keys())

    # Create input tensors
    x = torch.randn(batch_size, seq_len, embed_dim)
    self_attn_padding_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)

    # Forward pass
    y_tgt = layer_tgt(x, self_attn_padding_mask)

    x_permute = einops.rearrange(x, "b l d -> l b d")
    y_src_permute, _ = layer_src(
        x=x_permute,
        self_attn_mask=None,
        self_attn_padding_mask=self_attn_padding_mask,
    )
    y_src = einops.rearrange(y_src_permute, "l b d -> b l d")

    # Compare outputs
    print(f"Source Output: {y_src}")
    print(f"Target Output: {y_tgt}")

    # Create a mask for non-NaN values
    non_nan_mask = ~torch.isnan(y_src)

    # Apply the mask to both source and target outputs
    y_src_non_nan = y_src[non_nan_mask]
    y_tgt_non_nan = y_tgt[non_nan_mask]

    print(f"Source Output (non-NaN): {y_src_non_nan}")
    print(f"Target Output (non-NaN): {y_tgt_non_nan}")
    print(f"Max abs diff: {(y_src_non_nan - y_tgt_non_nan).abs().max()}")
    assert torch.allclose(y_src_non_nan, y_tgt_non_nan, atol=1e-6, rtol=1e-4)
