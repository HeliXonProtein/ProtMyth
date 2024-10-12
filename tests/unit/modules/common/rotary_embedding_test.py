# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

import pytest
import torch
from jaxtyping import Float, Bool
from typing import List

from protmyth.modules.common.attentions import RotaryEmbedding


@pytest.mark.parametrize(
    "c, n_head, batch_dims, q_len, kv_len",
    [
        (8, 2, [3], 5, 5),
        (4, 4, [2, 2, 2], 3, 8),
    ],
)
def test_rotary_embedding(
        c: int, n_head: int, batch_dims: List[int], q_len: int, kv_len: int
) -> None:
    """Test the make_graph method of the RotaryEmbedding module.

    Builds an RoPE PE module and tests its ability to process input tensor shapes effectively.

    Parameters
    ----------
    c : int
        Size of channels per head.
    n_head : int
        The number of attention heads.
    batch_dims : List[int]
        Dimensions representing batch size.
    q_len : int
        The length of the query.
    kv_len : int
        The length of the key/value.
    """
    device = torch.device('cpu')
    module = RotaryEmbedding(dim=c)

    # Generate random data for queries, keys
    # Note! We have split q_data, and kv_data into multiple heads
    q_data: Float[torch.Tensor, "batch_dims q_len n_head c"] = torch.randn(*batch_dims, q_len, n_head, c, device=device)
    k_data: Float[torch.Tensor, "batch_dims kv_len n_head c"] = torch.randn(*batch_dims, kv_len, n_head, c, device=device)

    # Perform the forward pass and capture the output
    output = module.forward(q_data, k_data)

    # Determine the expected output shape
    expected_shape_q = (*batch_dims, q_len, n_head, c)

    # # Assert the shape of the output
    assert output[0].shape == expected_shape_q, f"Expected shape {expected_shape_q}, but got {output[0].shape}"
