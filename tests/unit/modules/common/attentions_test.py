# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

import pytest
import torch
from jaxtyping import Float, Bool
from typing import List

from protmyth.modules.common.attentions import Attention, RotaryEmbedding


@pytest.mark.parametrize(
    "q_dim, kv_dim, c, n_head, out_dim, use_bias, gating, batch_dims, q_len, kv_len, use_rotary_embeddings",
    [
        (24, 32, 8, 2, 32, True, True, [3], 5, 5, False),
        (8, 6, 4, 4, 16, False, False, [2, 2, 2], 3, 8, False),
        (24, 32, 8, 2, 32, True, True, [3], 5, 5, True),
        (8, 6, 4, 4, 16, False, False, [2, 2, 2], 3, 8, True),
    ],
)
def test_attention(
        q_dim: int, kv_dim: int, c: int, n_head: int, out_dim: int, use_bias: bool,
        gating: bool, batch_dims: List[int], q_len: int, kv_len: int, use_rotary_embeddings: bool
) -> None:
    """Test the make_graph method of the Attention module.

    Builds an attention module and tests its ability to process input tensor shapes effectively.

    Parameters
    ----------
    q_dim : int
        The dimension of the query input.
    kv_dim : int
        The dimension of the key and value inputs.
    c : int
        The scaling factor or hidden dimension factor.
    n_head : int
        The number of attention heads.
    out_dim : int
        The dimension of the output.
    use_bias : bool
        Whether to use bias in linear transformations.
    gating : bool
        Whether to use gating mechanism.
    batch_dims : List[int]
        Dimensions representing batch size.
    q_len : int
        The length of the query.
    kv_len : int
        The length of the key/value.
    use_rotary_embeddings : bool
        Whether to use RoPE as the positional embedding.
    result_path : str
        Path to the directory where results are stored.
    """
    device = torch.device('cpu')
    module = Attention(
        q_dim=q_dim,
        kv_dim=kv_dim,
        c=c,
        n_head=n_head,
        out_dim=out_dim,
        use_bias=use_bias,
        gating=gating,
        use_rotary_embeddings=use_rotary_embeddings,
    )

    # Generate random data for queries, keys, values, and attention mask
    q_data: Float[torch.Tensor, "batch_dims q_len q_dim"] = torch.randn(*batch_dims, q_len, q_dim, device=device)
    kv_data: Float[torch.Tensor, "batch_dims kv_len kv_dim"] = torch.randn(*batch_dims, kv_len, kv_dim, device=device)
    attn_mask: Bool[torch.Tensor, "batch_dims q_len kv_len"] = torch.randint(0, 2, (*batch_dims, q_len, kv_len),
                                                                             device=device).bool()

    # Perform the forward pass and capture the output
    output = module.forward(q_data, kv_data, attn_mask=attn_mask)

    # Determine the expected output shape
    expected_shape = (*batch_dims, q_len, out_dim)

    # Assert the shape of the output
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"

    # test RoPE
    RoPE = RotaryEmbedding(dim=c)

    # Generate random data for queries, keys
    # Note! We have split q_data, and kv_data into multiple heads
    q4rope: Float[torch.Tensor, "batch_dims q_len n_head c"] = torch.randn(*batch_dims, q_len, n_head, c, device=device)
    k4rope: Float[torch.Tensor, "batch_dims kv_len n_head c"] = torch.randn(*batch_dims, kv_len, n_head, c,
                                                                            device=device)

    # Perform the forward pass and capture the output
    rope_q, _ = RoPE.forward(q4rope, k4rope)

    expected_shape_q = (*batch_dims, q_len, n_head, c)

    # # Assert the shape of the output
    assert rope_q.shape == expected_shape_q, f"Expected shape {expected_shape_q}, but got {rope_q.shape}"
