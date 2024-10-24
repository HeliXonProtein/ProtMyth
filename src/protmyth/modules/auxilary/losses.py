# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Auxilary losses like bert loss
"""

import torch
from torch import nn

from jaxtyping import Float

from protmyth.modules.common import mask


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
