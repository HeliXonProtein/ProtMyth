# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Auxilary losses like bert loss
"""

import torch
from torch import nn
import torch.nn.functional as F

import math
from jaxtyping import Float

from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module
from protmyth.modules.common import mask


@register_module("losses")
class RobertaLMHead(BaseModule[Float[torch.Tensor, "..."]]):
    """Head for masked language modeling."""

    def __init__(
            self,
            weight: Float[torch.Tensor, "..."],
            embed_dim: int = 1280,
            output_dim: int = 33,
            ) -> None:
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(
            self,
            features: Float[torch.Tensor, "..."]
            ) -> Float[torch.Tensor, "..."]:
        """Implementation of the forward function.

        Parameters
        ----------
        features: Float[torch.Tensor, "..."]
            A tensor containing the features.

        Returns
        -------
        torch.Tensor
            A tensor after robertahead
        """
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


def gelu(
        x: torch.Tensor
        ) -> torch.Tensor:
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    Parameters
    ----------
    x: Float[torch.Tensor, "...Z f_dim"]
        A tensor containing the features.

    Returns
    -------
    torch.Tensor
        A tensor after gelu
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def apc(
    x: torch.Tensor,
) -> torch.Tensor:
    """Perform average product correction, used for contact prediction.

    Parameters
    ----------
    x: Float[torch.Tensor, "...Z f_dim"]
        A tensor containing the features.

    Returns
    -------
    torch.Tensor
        A tensor after apc
    """
    a1 = torch.sum(x, -1, keepdim=True)
    a2 = torch.sum(x, -2, keepdim=True)
    a12 = torch.sum(x, (-1, -2), keepdim=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


def symmetrize(
        x: Float[torch.Tensor, "..."]
        ) -> Float[torch.Tensor, "..."]:
    """Make layer symmetric in final two dimensions, used for contact prediction.

    Parameters
    ----------
    x: Float[torch.Tensor, "...Z f_dim"]
        A tensor containing the features.

    Returns
    -------
    torch.Tensor
        A tensor after symmetrize
    """
    return x + x.transpose(-1, -2)


@register_module("losses")
class ContactPredictionHead(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
            self,
            in_features: int,
            prepend_bos: bool,
            append_eos: bool,
            bias: bool,
            eos_idx: int,
            ) -> None:
        super().__init__()
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
        """Implementation of the forward function.
        Parameters
        ----------
        tokens: Float[torch.Tensor, "...Z f_dim"]
            token containing the features.
        attentions: Float[torch.Tensor, "...Z f_dim"]
            attentions containing the weights.

        Returns
        -------
        torch.Tensor
            A tensor after ContactPredictionHead
        """
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


def compute_RobertaLMHead_loss(
        logits: Float[torch.Tensor, "..."],
        label: Float[torch.Tensor, "..."],
        label_mask: Float[torch.Tensor, "..."],
        ) -> Float[torch.Tensor, "..."]:
    """compute mask LM loss.

    Parameters
    ----------
    logits: Float[torch.Tensor, "..."]
        A tensor containing the logits.
    label: Float[torch.Tensor, "..."]
        A tensor containing the label.
    label_mask: Float[torch.Tensor, "..."]
        A tensor containing the label_mask.
    Returns
    -------
    torch.Tensor
        A tensor of batch loss
    """

    # CrossEntropyLoss
    with torch.no_grad():
        label = label.long()
        label = torch.where(label_mask == 1, label, (torch.ones_like(label) * (-100)).type_as(label))
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fn(logits, label)
    loss = mask.masked_mean(value=loss, mask=label_mask, dim=(-1, -2))
    return loss
