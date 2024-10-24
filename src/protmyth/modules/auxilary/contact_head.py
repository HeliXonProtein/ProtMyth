# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Auxilary losses like bert loss
"""

import torch
from torch import nn

from jaxtyping import Float

from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module


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


@register_module("auxilary")
class ContactPredictionHead(BaseModule[Float[torch.Tensor, "..."]]):
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
