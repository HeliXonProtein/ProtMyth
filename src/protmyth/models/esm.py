# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""ESM models using protmyth modules.

Include:
1. ESM2
"""



from typing import Union, List, Dict, Any
import torch
from torch import nn
from jaxtyping import Float, PyTree, Shaped, Int
import einops


from protmyth.models.base import BaseModel
from protmyth.modules.seqencoders.esm.transformer import TransformerLayer
from protmyth.modules.auxilary.losses import RobertaLMHead


class ESM2(BaseModel[
               PyTree[Int[torch.Tensor, 'batch seq_len']],
               Float[torch.Tensor, 'batch seq_len embed_dim'],
               Float[torch.Tensor, "..."],
           ]):

    """
    A model for the ESM2 architecture using transformer layers.

    Parameters
    ----------
    num_layers : int, optional
        Number of transformer layers, by default 33.
    embed_dim : int, optional
        Dimension of the embeddings, by default 1280.
    attention_heads : int, optional
        Number of attention heads, by default 20.
    alphabet : Union[esm.data.Alphabet, str], optional
        Alphabet for tokenization, by default "ESM-1b".
    token_dropout : bool, optional
        Whether to apply token dropout, by default True.
    """

    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet_size: int = 23,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.alphabet_size = alphabet_size
        self.padding_idx = 21
        self.mask_idx = 22

        self._init_submodules()

    def _init_submodules(self) -> None:
        """
        Initialize submodules for the ESM2 model.
        """
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.axial_transformer_stacks = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.emb_layer_norm_after = nn.LayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(
        self,
        get_loss: bool = False,
        batch: PyTree[Int[torch.Tensor, 'batch seq_len']],
    ) -> Union[Float[torch.Tensor, 'batch seq_len embed_dim']]:
        """
        Perform a forward pass through the ESM2 model.

        Parameters
        ----------
        batch: dict of batch samples
            tokens : Int[torch.Tensor, "batch seq_len"]
                Input token indices.

        Returns
        -------
        """
        need_head_weights = True

        seq_tokens = batch['seq_tokens']

        padding_mask = seq_tokens.eq(self.padding_idx)  # B, T

        if not padding_mask.any():
            padding_mask = None

        seq_embedding = self.embed_scale * self.embed_tokens(seq_tokens)

        # Use einops for transpose operation
        seq_embedding = einops.rearrange(seq_embedding, 'b t e -> t b e')

        for layer_idx, layer in enumerate(self.layers):
            seq_embedding, attn = layer(
                seq_embedding,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )

        seq_embedding = self.emb_layer_norm_after(seq_embedding)
        seq_embedding = einops.rearrange(seq_embedding, 't b e -> b t e')
        seq_embedding = self.lm_head(seq_embedding)

        return seq_embedding


