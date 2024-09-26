# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Auxilary losses like bert loss
"""

import torch
from torch import nn
import torch.nn.functional as F

import losses.utils as utils

from jaxtyping import Float, Bool
from typing import Optional
import torchviz
from graphviz import Digraph
import einops
from collections.abc import Sequence

from protmyth.modules import common, base, register

def distogram_prediction_loss(
    logits: Tensor,
    distogram_fn: Callable[[Tensor], Tensor],
    pos_cb_gt: Tensor,
    pos_cb_mask: Tensor,
    pair_mask: Tensor | None = None,
) -> Tensor:
    """Compute distogram prediction loss.

    See alphafold supplementary 1.9.8 for details.

    Args:
        logits: Predicted logits of distogram bins, (B, N, N, num_bins).
        distogram_fn: Callable that converts distance matrix to distogram, e.g. DistanceToBins.
        pos_cb_gt: Ground truth CB positions, (B, N, 3).
        pos_cb_mask: Masks of ground truth CBs, (B, N).
        pair_mask: Additional masks of CB pairs. None or (B, N, N).

    Returns:
        Distogram loss, (B).
    """
    pair_dist = common.geometric.cdist(pos_cb_gt, pos_cb_gt)
    pair_bin = distogram_fn(pair_dist)
    logits = einops.rearrange(logits, "... i j d -> ... d i j ")
    loss = F.cross_entropy(logits, pair_bin, reduction="none")

    if pair_mask is None:
        pair_mask = common.mask.node_mask_to_pair_mask(pos_cb_mask)
    else:
        pair_mask = pair_mask * common.mask.node_mask_to_pair_mask(pos_cb_mask)
    loss = common.mask.masked_mean(loss, pair_mask, dim=(-1, -2))
    return loss


def masked_msa_loss(
    logits: Tensor,
    msa_gt: torch.LongTensor,
    bert_mask: torch.BoolTensor,
) -> Tensor:
    """
    Computes BERT-style masked MSA loss.

    See alphafold supplementary subsection 1.9.9.

    Args:
        logits: Predicted logits of residue distribution, (B, M, N, d_class).
                d_class = 23 for monomer d_class = 22 for multimer.
                The last dim for d_class = 23 is mask dim which will not be predicted.
        msa_gt: Ground truth MSA, (B, M, N).
        bert_mask: Of where gt msa is masked, (B, M, N).
    Returns:
        Masked MSA loss, (B).
    """
    logits = einops.rearrange(logits, "... s i d -> ... d s i ")
    loss = F.cross_entropy(logits, msa_gt, reduction="none")
    loss = common.mask.masked_mean(loss, bert_mask, dim=(-1, -2))
    return loss
