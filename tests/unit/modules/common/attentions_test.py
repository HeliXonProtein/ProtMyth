# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Unit tests for the Attentions.
"""

import pytest
import torch

from protmyth.modules.common.attentions import Attention


@pytest.mark.parametrize(
    "q_dim, kv_dim, c, n_head, out_dim, use_bias, gating, batch_dims, q_len, kv_len, result_path",
    [
        (24, 32, 8, 2, 32, True, True, [3], 5, 5,
            "/datapool/data3/storage/public/ProtMythItems/graphs/common/attention_1"),
        (8, 6, 4, 4, 16, False, False, [2, 2, 2], 3, 8,
            "/datapool/data3/storage/public/ProtMythItems/graphs/common/attention_2"),
    ],
)
def test_attention_graph(
    q_dim, kv_dim, c, n_head, out_dim, use_bias, gating, batch_dims, q_len, kv_len, result_path
):
    """Test the make_graph method.

    Build an attention module and tests its graph.
    """
    module = Attention(
        q_dim=q_dim,
        kv_dim=kv_dim,
        c=c,
        n_head=n_head,
        out_dim=out_dim,
        use_bias=use_bias,
        gating=gating,
    )
    dot = module.make_graph(
        batch_dims=batch_dims,
        q_len=q_len,
        kv_len=kv_len,
        device=torch.device("cpu"),
    )
    dot.format = 'png'
    dot.render(result_path)
