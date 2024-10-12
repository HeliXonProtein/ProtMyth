# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

import pytest
import torch
from protmyth.modules.seqencoders.esm.transformer_3 import TransformerLayer


@pytest.fixture
def transformer_without_rope() -> TransformerLayer:
    """
    Fixture to create a TransformerLayer without RoPE.

    Returns
    -------
    TransformerLayer
        Transformer layer without RoPE.
    """
    return TransformerLayer(
        embed_dim=128,
        ffn_embed_dim=256,
        attention_heads=4,
        use_rotary_embeddings=False
    )


@pytest.fixture
def transformer_with_rope() -> TransformerLayer:
    """
    Fixture to create a TransformerLayer with RoPE.

    Returns
    -------
    TransformerLayer
        Transformer layer with RoPE.
    """
    return TransformerLayer(
        embed_dim=128,
        ffn_embed_dim=256,
        attention_heads=4,
        use_rotary_embeddings=True
    )


def test_transformer_layer_forward_without_rope(transformer_without_rope: TransformerLayer) -> None:
    """
    Test forward pass of TransformerLayer without RoPE.

    Parameters
    ----------
    transformer_without_rope : TransformerLayer
        TransformerLayer instance without RoPE.

    Asserts
    -------
    AssertionError
        If output shape does not match expected dimensions.
    """
    batch_size, seq_len, embed_dim = 8, 16, 128
    x = torch.randn(batch_size, seq_len, embed_dim)
    positions = torch.randn(batch_size, seq_len, 2)

    # Run the forward pass
    output, _ = transformer_without_rope(x, positions)

    # Assert output shape
    assert output.shape == (batch_size, seq_len, embed_dim), "Output shape mismatch."


def test_transformer_layer_forward_with_rope(transformer_with_rope: TransformerLayer) -> None:
    """
    Test forward pass of TransformerLayer with RoPE.

    Parameters
    ----------
    transformer_with_rope : TransformerLayer
        TransformerLayer instance with RoPE.

    Asserts
    -------
    AssertionError
        If output shape does not match expected dimensions.
    """
    batch_size, seq_len, embed_dim = 8, 16, 128
    x = torch.randn(batch_size, seq_len, embed_dim)
    positions = torch.randn(batch_size, seq_len, 2)

    # Run the forward pass
    output, _ = transformer_with_rope(x, positions)

    # Assert output shape
    assert output.shape == (batch_size, seq_len, embed_dim), "Output shape mismatch."


def test_rope_changes_query_key(transformer_with_rope: TransformerLayer) -> None:
    """
    Test that RoPE modifies query and key tensors.

    Parameters
    ----------
    transformer_with_rope : TransformerLayer
        TransformerLayer instance with RoPE.

    Asserts
    -------
    AssertionError
        If RoPE doesn't modify query and key tensors.
    """
    batch_size, seq_len, embed_dim = 8, 16, 128
    x = torch.randn(batch_size, seq_len, embed_dim)
    # positions = torch.randn(batch_size, seq_len, 2)

    # Apply RoPE
    transformer = transformer_with_rope
    q_rot, k_rot = transformer._apply_rope_to_qk(x, x)

    # Assert the shapes of modified queries and keys
    assert q_rot.shape == (batch_size, seq_len, embed_dim), "RoPE Q shape mismatch."
    assert k_rot.shape == (batch_size, seq_len, embed_dim), "RoPE K shape mismatch."

    # Ensure that RoPE modifies the tensors
    assert not torch.allclose(q_rot, x), "RoPE did not modify query."
    assert not torch.allclose(k_rot, x), "RoPE did not modify key."


def test_attention_with_mask(transformer_without_rope: TransformerLayer) -> None:
    """
    Test that attention with mask is applied correctly.

    Parameters
    ----------
    transformer_without_rope : TransformerLayer
        TransformerLayer instance without RoPE.

    Asserts
    -------
    AssertionError
        If output shape doesn't match with attention mask.
    """
    batch_size, seq_len, embed_dim = 8, 16, 128
    x = torch.randn(batch_size, seq_len, embed_dim)
    positions = torch.randn(batch_size, seq_len, 2)

    # Create a random attention mask (batch, seq_len, seq_len)
    attn_mask = torch.randint(0, 2, (batch_size, seq_len, seq_len), dtype=torch.bool)

    # Run forward pass with the mask
    output, _ = transformer_without_rope(x, positions, self_attn_mask=attn_mask)

    # Assert that output shape matches input
    assert output.shape == (batch_size, seq_len, embed_dim), "Output shape mismatch with attention mask."


def test_transformer_layer_with_padding_mask(transformer_with_rope: TransformerLayer) -> None:
    """
    Test the TransformerLayer with a padding mask.

    Parameters
    ----------
    transformer_with_rope : TransformerLayer
        TransformerLayer instance with RoPE.

    Asserts
    -------
    AssertionError
        If output shape doesn't match with padding mask.
    """
    batch_size, seq_len, embed_dim = 8, 16, 128
    x = torch.randn(batch_size, seq_len, embed_dim)
    positions = torch.randn(batch_size, seq_len, 2)

    # Create padding mask (batch, seq_len)
    padding_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)

    # Run the forward pass with padding mask
    output, _ = transformer_with_rope(x, positions, self_attn_padding_mask=padding_mask)

    # Assert output shape
    assert output.shape == (batch_size, seq_len, embed_dim), "Output shape mismatch with padding mask."


def test_forward_with_need_head_weights(transformer_without_rope: TransformerLayer) -> None:
    """
    Test the case where attention head weights are requested.

    Parameters
    ----------
    transformer_without_rope : TransformerLayer
        TransformerLayer instance without RoPE.

    Asserts
    -------
    AssertionError
        If output shape doesn't match when attention head weights are requested.
    """
    batch_size, seq_len, embed_dim = 8, 16, 128
    x = torch.randn(batch_size, seq_len, embed_dim)
    positions = torch.randn(batch_size, seq_len, 2)

    # Run forward pass with need_head_weights=True
    output, attn_weights = transformer_without_rope(x, positions, need_head_weights=True)

    # Assert output shape
    assert output.shape == (batch_size, seq_len, embed_dim), "Output shape mismatch when head weights are requested."
    assert attn_weights is not None, "Attention weights should be returned when requested."
