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

import math
from jaxtyping import Float, Bool
from typing import Optional
import torchviz
from graphviz import Digraph
import einops
from collections.abc import Sequence

from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module


@register_module("losses")
class BertHead(BaseModule[Float[torch.Tensor, "..."]]):
    """Head for masked language modeling."""

    def __init__(
            self,
            in_features: int=1280,
            out_features: int=33,
    ) -> None:
        super(BertHead, self).__init__()
        self.linear1 = nn.Linear(in_features, in_features)
        self.layer_norm = nn.LayerNorm(in_features)
        self.linear2 = nn.Linear(in_features, out_features)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def forward(
            self, 
            esm_embedding: Float[torch.Tensor, "... Z z_dim"], 
            gt_esm: Float[torch.Tensor, "... Z #z_dim"], 
            esm_bert_mask: Float[torch.Tensor, "... Z #z_dim"],
            get_embeds: bool=False,
    ) -> float:
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


@register_module("losses")
class RobertaLMHead(BaseModule[Float[torch.Tensor, "..."]]):
    """Head for masked language modeling."""

    def __init__(
            self, 
            embed_dim: int=1280,
            output_dim: int=33,
            weight: Float[torch.Tensor, "..."]=None
        ) -> None:
        super(RobertaLMHead, self).__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(
            self, 
            features: Float[torch.Tensor, "...Z f_dim"]
        ) -> Float[torch.Tensor, "... Z w_dim"]:
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@register_module("losses")
class MSAMaskPredictHead(BaseModule[Float[torch.Tensor, "..."]]):
    def __init__(
            self,
            in_features: int=256,
            out_features: int=23,
    ) -> None:
        super(MSAMaskPredictHead, self).__init__()
        self.proj = nn.Linear(in_features, out_features, initializer='zeros')
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def forward(
            self, 
            esm_embedding: Float[torch.Tensor, "... Z z_dim"], 
            gt_esm: Float[torch.Tensor, "... Z #z_dim"], 
            bert_mask: Float[torch.Tensor, "... Z #z_dim"],
            get_embeds: bool=False,
    ) -> float:
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


def apc(
    x:  Float[torch.Tensor, "..."]
    ) ->  Float[torch.Tensor, "..."]:
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


def symmetrize(
        x: Float[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "..."]:
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


@register_module("losses")
class ContactPredictionHead(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        prepend_bos: bool,
        append_eos: bool,
        bias=True,
        eos_idx: Optional[int] = None,
    ) -> None:
        super(ContactPredictionHead).__init__()
        self.in_features = in_features
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        if append_eos and eos_idx is None:
            raise ValueError("Using an alphabet with eos token, but no eos token was passed in.")
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1, bias)
        self.activation = nn.Sigmoid()

    def forward(
            self,
            tokens: Float[torch.Tensor, "..."],
            attentions: Float[torch.Tensor, "..."]
        ) -> Float[torch.Tensor, "..."]:
        # remove eos token attentions
        if self.append_eos:
            eos_mask = tokens.ne(self.eos_idx).to(attentions)
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: B x C x T x T
        attentions = attentions.to(
            self.regression.weight.device
        )  # attentions always float32, may need to convert to float16
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        return self.activation(self.regression(attentions).squeeze(3))