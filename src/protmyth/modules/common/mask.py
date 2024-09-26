# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Common mask utilities to perform masked tensor operations.
"""

from typing import Optional, Union, Tuple
import torch
import einops
from torch import Tensor


# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Common mask utilities to perform masked tensor operations.  
"""

from typing import Union, Tuple
import torch
import einops
from jaxtyping import Float


def node_mask_to_pair_mask(
        mask_1d: Float[torch.Tensor, "... N"]
) -> Float[torch.Tensor, "... N N"]:
    """Compute pair mask from node mask.

    Parameters
    ----------
    mask_1d : Float[torch.Tensor, "... N"]
        Node mask of shape (..., N).

    Returns
    -------
    Float[torch.Tensor, "... N N"]
        Pair mask with shape (..., N, N).
    """
    return einops.einsum(mask_1d, mask_1d, '... i, ... j -> ... i j')


def masked_divide(
        value: Float[torch.Tensor, "..."],
        mask: Float[torch.Tensor, "..."],
        mask_summed: Float[torch.Tensor, "..."],
        eps: float = 1e-10
) -> Float[torch.Tensor, "..."]:
    """Divide `value` by `mask_summed` element-wise, scaled by `mask`.

    Parameters
    ----------
    value : Float[torch.Tensor, "..."]
        Tensor to be divided.
    mask : Float[torch.Tensor, "..."]
        Binary mask of the same shape as `value`.
    mask_summed : Float[torch.Tensor, "..."]
        Sum of mask along the specified dimensions.
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    Float[torch.Tensor, "..."]
        Resulting tensor after masked division.
    """
    return value * mask / (mask_summed + eps)


def masked_mean(
        value: Float[torch.Tensor, "..."],
        mask: Float[torch.Tensor, "..."],
        dim: Union[int, Tuple[int, ...]] = -1,
        eps: float = 1e-10,
        return_masked: bool = False,
) -> Union[Float[torch.Tensor, "..."],
           Tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]]:
    """Compute the masked mean of a tensor.

    Parameters
    ----------
    value : Float[torch.Tensor, "..."]
        Tensor from which to compute the mean.
    mask : Float[torch.Tensor, "..."]
        Binary mask of the same shape as `value`.
    dim : int or tuple of int, optional
        Dimensions along which the mean is computed.
    eps : float, optional
        Small constant for numerical stability.
    return_masked : bool, optional
        Whether to return the masked value alongside the mean.

    Returns
    -------
    Float[torch.Tensor, "..."] or Tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]
        Masked mean tensor, and optionally the masked tensor itself.
    """
    if value.ndim != mask.ndim:
        raise ValueError("`value` and `mask` must have the same number of dimensions.")

    if isinstance(dim, int):
        dim = (dim,)

    masked_value = masked_divide(value, mask, mask.sum(dim, keepdim=True), eps)
    mean_value = masked_value.sum(dim)

    if return_masked:
        return mean_value, masked_value

    return mean_value


def masked_quadratic_mean(
        value: Float[torch.Tensor, "..."],
        mask: Float[torch.Tensor, "..."],
        dim: Union[int, Tuple[int, ...]] = -1,
        eps: float = 1e-10,
        return_masked: bool = False,
) -> Union[Float[torch.Tensor, "..."], Tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]]:
    """Compute the masked quadratic mean of a tensor.

    Parameters
    ----------
    value : Float[torch.Tensor, "..."]
        Tensor for which to compute the quadratic mean.
    mask : Float[torch.Tensor, "..."]
        Binary mask of the same shape as `value`.
    dim : int or tuple of int, optional
        Dimensions along which the quadratic mean is computed.
    eps : float, optional
        Small constant for numerical stability.
    return_masked : bool, optional
        Whether to return the masked value alongside the quadratic mean.

    Returns
    -------
    Float[torch.Tensor, "..."] or Tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]
        Masked quadratic mean tensor, and optionally the masked tensor itself.
    """
    if value.ndim != mask.ndim:
        raise ValueError("`value` and `mask` must have the same number of dimensions.")

    masked_value = value * mask
    mean_value = torch.sqrt((masked_value ** 2).sum(dim) / (mask.sum(dim) + eps))

    if return_masked:
        return mean_value, masked_value

    return mean_value


def masked_max(
        value: Float[torch.Tensor, "..."],
        mask: Float[torch.Tensor, "..."],
        dim: int = -1
) -> Float[torch.Tensor, "..."]:
    """Compute the maximum of a tensor masked with a binary mask.

    Parameters
    ----------
    value : Float[torch.Tensor, "..."]
        Tensor from which to compute the masked maximum.
    mask : Float[torch.Tensor, "..."]
        Binary mask of the same shape as `value`.
    dim : int, optional
        Dimension along which to compute the maximum.

    Returns
    -------
    Float[torch.Tensor, "..."]
        Tensor of masked maximum values.
    """
    masked_value = torch.where(mask.bool(), value, torch.tensor(float("-inf"), dtype=value.dtype, device=value.device))
    return masked_value.max(dim).values