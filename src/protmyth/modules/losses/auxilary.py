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

class BertHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(
            self,
            in_features=1280,
            out_features=33):
        super(BertHead, self).__init__()
        self.linear1 = nn.Linear(in_features, in_features)
        self.layer_norm = nn.LayerNorm(in_features)
        self.linear2 = nn.Linear(in_features, out_features)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def forward(self, esm_embedding, gt_esm, esm_bert_mask, get_embeds=False):
        with torch.no_grad():
            gt_esm = gt_esm.long()
            gt_esm = torch.where(esm_bert_mask == 1, gt_esm, (torch.ones_like(gt_esm) * (-100)).type_as(gt_esm))

        logits = nn.GELU(self.linear1(esm_embedding))
        logits = self.layer_norm(logits)
        logits = self.linear2(logits)
        loss = self.loss_fn(logits.permute(0, 3, 1, 2).contiguous(), gt_esm) # nn.CrossEntropyLoss assumes the logits at dim 1
        loss = utils.mask_mean(value=loss, mask=esm_bert_mask, dim=[-1, -2])

        if get_embeds:
            mask_pred = logits.softmax(dim=-1)
            return loss, mask_pred
        else:
            return loss


class MSAMaskPredictHead(nn.Module):
    def __init__(
            self,
            in_features=256,
            out_features=23):
        super(MSAMaskPredictHead, self).__init__()
        self.proj = nn.Linear(in_features, out_features, initializer='zeros')
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def forward(self, msa_embedding, gt_msa, bert_mask, get_embeds=False):
        """

        :param msa_embedding: torch.float32: [batch_size, num_clusters, L, 23]
        :param gt_msa: torch.int: [batch_size, num_clusters, L]
        :param bert_mask: torch.bool: [batch_size, num_clusters, L], bert_mask=1 means this slot needs prediction
        :return: loss
        """
        with torch.no_grad():
            gt_msa = gt_msa.long()
            gt_msa = torch.where(bert_mask == 1, gt_msa, (torch.ones_like(gt_msa) * (-100)).type_as(gt_msa)) # masked gt

        # to reduce templates from msa
        msa_embedding = msa_embedding[:, :gt_msa.shape[1]]

        logits = self.proj(msa_embedding)
        loss = self.loss_fn(logits.permute(0, 3, 1, 2).contiguous(), gt_msa) # nn.CrossEntropyLoss assumes the logits at dim 1
        loss = utils.mask_mean(value=loss, mask=bert_mask, dim=[-1, -2])

        if get_embeds:
            mask_pred = logits.softmax(dim=-1)
            return loss, mask_pred
        else:
            return loss