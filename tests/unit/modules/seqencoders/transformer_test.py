# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

import pytest
import torch
from protmyth.modules.seqencoders.esm.transformer import TransformerLayer


@pytest.mark.parametrize(
    "embed_dim, ffn_embed_dim, attention_heads, add_bias_kv,"
    "use_esm1b_layer_norm, use_rotary_embeddings, gating, batch_size, seq_len",
    [
        (64, 128, 8, True, False, False, True, 2, 10),
        (128, 256, 16, False, True, True, False, 4, 20),
        (256, 512, 32, True, False, True, True, 8, 30),
    ]
)
def test_transformer_layer(
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
    layer = TransformerLayer(
        embed_dim=embed_dim,
        ffn_embed_dim=ffn_embed_dim,
        attention_heads=attention_heads,
        add_bias_kv=add_bias_kv,
        use_esm1b_layer_norm=use_esm1b_layer_norm,
        use_rotary_embeddings=use_rotary_embeddings,
        gating=gating,
    )

    # Create input tensors
    x = torch.randn(batch_size, seq_len, embed_dim)
    self_attn_mask = torch.randint(0, 2, (batch_size, seq_len, seq_len), dtype=torch.bool)
    self_attn_padding_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)

    # Forward pass
    output = layer(x, self_attn_mask, self_attn_padding_mask)

    # Verify output shape
    assert output.shape == (batch_size, seq_len, embed_dim), f"Output shape mismatch: \
            expected {(batch_size, seq_len, embed_dim)}, got {output.shape}"

    # Ensure no NaN values in the output
    assert torch.isnan(output).sum() == 0, "Output contains NaN values"
