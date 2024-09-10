# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""seq2pair embeddings
"""

import torch
from torch import nn
import torch.nn.functional as F

class Relative_Positional_Embedding(nn.Module):
    def __init__(self, att_embed_dim, relpos_len=32):
        super(Relative_Positional_Embedding, self).__init__()
        self.relpos_len = relpos_len
        self.pair_activations = nn.Linear(2 * relpos_len + 1, att_embed_dim)

    def forward(self, Z):
        with torch.no_grad():
            B, L = Z.shape[:2]
            di = torch.arange(0, L, layout=Z.layout, device=Z.device).unsqueeze(0)
            dj = torch.arange(0, L, layout=Z.layout, device=Z.device).unsqueeze(1)
            d = di - dj
            d[d > self.relpos_len] = self.relpos_len
            d[d < -self.relpos_len] = -self.relpos_len
            d = 32 - d  # to keep same with alphafold

            relpos_onehot = torch.eye(2 * self.relpos_len + 1)[d]
            relpos_feat = relpos_onehot.expand(B, -1, -1, -1).contiguous()
        return self.pair_activations(relpos_feat)


class PairEmbedder(nn.Module):
    def __init__(self, n_head, att_embed_dim):
        super(PairEmbedder, self).__init__()
        self.layernorm = nn.LayerNorm(n_head)
        self.linear = nn.Linear(n_head, att_embed_dim)
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, pair):
        z = self.layernorm(pair)
        z = self.linear(z)
        z = self.dropout(z)
        return z