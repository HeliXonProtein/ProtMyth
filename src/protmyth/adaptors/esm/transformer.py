# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Esm Transformers mapper
"""

import torch
from protmyth.adaptors.mapper import ModuleMapper
from protmyth.adaptors.common import linear_mapper
from typing import Dict
from jaxtyping import Float


def attention_mapper(tgt_pfx: str = "", src_pfx: str = "") -> ModuleMapper:
    """Creates a module mapper for transformer layers.

    Parameters
    ----------
    tgt_pfx : str, optional
        Target prefix for module mapping, by default "".
    src_pfx : str, optional
        Source prefix for module mapping, by default "".

    Returns
    -------
    ModuleMapper
        A configured ModuleMapper object for transformer layers.
    """

    def _concat_kv_weights(to_concat_dict: Dict[str, Float[torch.Tensor, "..."]], _: torch.Tensor) -> torch.Tensor:
        """Concatenates key-value weights along the first dimension.

        Parameters
        ----------
        to_concat_dict : dict
            Dictionary containing tensors to concatenate.
        _ : torch.Tensor
            Placeholder tensor, not used in the function.

        Returns
        -------
        torch.Tensor
            Concatenated tensor.
        """
        return torch.cat(list(to_concat_dict.values()), dim=0)

    return (
        ModuleMapper(tgt_pfx, src_pfx)
        .add_submodule(linear_mapper("q_linear", "q_proj"))
        .add_multimap("kv_linear.weight", ["k_proj.weight", "v_proj.weight"], _concat_kv_weights)
        .add_multimap("kv_linear.bias", ["k_proj.bias", "v_proj.bias"], _concat_kv_weights)
        .add_submodule(linear_mapper("output_linear", "out_proj"))
        .add_submodule(rotary_embedding_mapper("rot_emb", "rot_emb"))
    )
