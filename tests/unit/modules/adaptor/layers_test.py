# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

import pytest
import torch
import einops
import esm

from protmyth.modules.common.attentions import Attention, RotaryEmbedding
from protmyth.adaptors.esm.layers import attention_mapper, rotary_embedding_mapper


@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim, attention_heads, if_attn_padding_mask",
    [
        (1, 7, 8, 2, 0),
        (2, 8, 16, 4, 1),
        (3, 3, 24, 6, 0),
        (4, 2, 30, 5, 1),
    ]
)
def test_attention_mapper(
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    attention_heads: int,
    if_attn_padding_mask: bool
) -> None:
    """Test the attention mapper between source and target attention layers with random parameters.

    Parameters
    ----------
    batch_size : int
        Number of samples per batch.
    seq_len : int
        Length of each sequence.
    embed_dim : int
        Dimensionality of the embedding space.
    attention_heads : int
        Number of attention heads.
    if_attn_padding_mask : bool
        Flag to use attention padding mask.

    Returns
    -------
    None
    """
    layer_src = esm.multihead_attention.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=attention_heads,
        add_bias_kv=False,
        add_zero_attn=False,
        use_rotary_embeddings=True,
    )
    layer_tgt = Attention(
        q_dim=embed_dim,
        kv_dim=embed_dim,
        c=embed_dim // attention_heads,
        n_head=attention_heads,
        out_dim=embed_dim,
        use_bias=True,
        gating=False,
        use_rotary_embeddings=True,
    )

    # Initialize source layer parameters
    for p in layer_src.parameters():
        p.data.normal_(0.0, 0.5)

    # Map states between the two layers
    mapper = attention_mapper(tgt_pfx="", src_pfx="")
    result = mapper(layer_tgt.state_dict(), layer_src.state_dict())
    assert set(result.matched_source) == set(layer_src.state_dict().keys())
    assert set(result.matched_target) == set(layer_tgt.state_dict().keys())

    # Generate random inputs
    x = torch.randn(batch_size, seq_len, embed_dim)

    self_attn_padding_mask = (
        torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)
        if if_attn_padding_mask else None
    )

    if self_attn_padding_mask is not None:
        attn_mask = ~einops.rearrange(self_attn_padding_mask, 'b l -> b 1 l')
    else:
        attn_mask = None

    # Compute attention outputs
    x_permute = einops.rearrange(x, "b l d -> l b d")

    y_tgt = layer_tgt(q_data=x, kv_data=x, attn_mask=attn_mask)

    y_src_permute, _ = layer_src(
        query=x_permute,
        key=x_permute,
        value=x_permute,
        key_padding_mask=self_attn_padding_mask,
        need_weights=True,
        need_head_weights=True,
        attn_mask=None,
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


@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim, n_heads",
    [
        (2, 8, 16, 4),
        (3, 12, 24, 6),
    ]
)
def test_rotary_embedding_mapper(batch_size: int, seq_len: int, embed_dim: int, n_heads: int) -> None:
    """Test the rotary embedding mapper between source and target rotary embeddings with random parameters.

    Parameters
    ----------
    seq_len : int
        Length of each sequence.
    embed_dim : int
        Dimensionality of the embedding space.
    n_heads : int
        Number of attention heads.

    Returns
    -------
    None
    """
    # Create source and target rotary embedding layers
    layer_src = esm.rotary_embedding.RotaryEmbedding(dim=embed_dim // n_heads)
    layer_tgt = RotaryEmbedding(dim=embed_dim // n_heads)

    # Initialize source layer parameters
    for param in layer_src.parameters():
        param.data.normal_(0.0, 0.5)

    # Assuming rotary_embedding_mapper can map states between the two layers
    mapper = rotary_embedding_mapper(tgt_pfx="", src_pfx="")
    result = mapper(layer_tgt.state_dict(), layer_src.state_dict())
    assert set(result.matched_source) == set(layer_src.state_dict().keys())
    assert set(result.matched_target) == set(layer_tgt.state_dict().keys())

    # Generate random q and k inputs
    q = torch.randn(batch_size, seq_len, n_heads, embed_dim // n_heads)
    k = torch.randn(batch_size, seq_len, n_heads, embed_dim // n_heads)

    q_src_input = einops.rearrange(q, "b s n c -> (b n) s c")
    k_src_input = einops.rearrange(k, "b s n c -> (b n) s c")

    # Compute rotary embeddings
    q_src_out, k_src_out = layer_src(q_src_input, k_src_input)
    q_src = einops.rearrange(q_src_out, "(b s) n c -> b n s c", b=batch_size)
    k_src = einops.rearrange(k_src_out, "(b s) n c -> b n s c", b=batch_size)

    q_tgt, k_tgt = layer_tgt(q, k)

    # Compare outputs
    print(f"Max abs diff Q: {(q_src - q_tgt).abs().max()}")
    print(f"Max abs diff K: {(k_src - k_tgt).abs().max()}")

    assert torch.allclose(q_src, q_tgt, atol=1e-6, rtol=1e-4)
    assert torch.allclose(k_src, k_tgt, atol=1e-6, rtol=1e-4)
