# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

from typing import Union, List, Dict, Any
import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, PyTree, Shaped
import einops


from protmyth.models.base import BaseModel
from protmyth.modules.seqencoders.esm.axial_transformer import AxialTransformerLayer
from protmyth.modules.embeddings.seq2node import 
from protmyth.modules.auxilary.bert import BertHead


class ESM2(BaseModel[
               PyTree[Union[Shaped[torch.Tensor, '...']]],
               PyTree[Float[torch.Tensor, '...']],
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

        self.contact_head = ContactPredictHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = nn.LayerNorm(self.embed_dim)

        self.lm_head = BertHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(
        self,
        batch: PyTree[Union[Shaped[torch.Tensor, 'B L']]],
        get_loss: bool = False,
    ) -> PyTree[Float[torch.Tensor, '...']]:
        """
        Perform a forward pass through the ESM2 model.

        Parameters
        ----------
        batch: dict of batch samples
            tokens : Int[torch.Tensor, "batch seq_len"]
                Input token indices.
            repr_layers : List[int], optional
                Indices of layers to return hidden states for, by default [].
            need_head_weights : bool, optional
                Whether to return attention weights, by default False.
            return_contacts : bool, optional
                Whether to return contact predictions, by default False.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing logits, representations, attentions, and contacts.
        """
        need_head_weights = True

        seq_tokens = batch['seq_tokens']
        repr_layers = batch['repr_layers']

        padding_mask = seq_tokens.eq(self.padding_idx)  # B, T

        seq_embedding = self.embed_scale * self.embed_tokens(seq_tokens)

        repr_layers = set(repr_layers)
        hidden_embeddings = {}
        if 0 in repr_layers:
            hidden_embeddings[0] = seq_embedding

        if need_head_weights:
            attn_weights = []

        # Use einops for transpose operation
        seq_embedding = einops.rearrange(seq_embedding, 'b t e -> t b e')

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            seq_embedding, attn = layer(
                seq_embedding,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_embeddings[layer_idx + 1] = einops.rearrange(seq_embedding, 't b e -> b t e')
            if need_head_weights:
                attn_weights.append(einops.rearrange(attn, 'h b t1 t2 -> b h t1 t2'))

        seq_embedding = self.emb_layer_norm_after(seq_embedding)
        seq_embedding = einops.rearrange(seq_embedding, 't b e -> b t e')

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_embeddings[layer_idx + 1] = seq_embedding
        seq_embedding = self.lm_head(seq_embedding)

        result = {"seq_embedding": seq_embedding, "hidden_embeddings": hidden_embeddings}
        if need_head_weights:
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            # if return_contacts:
            #     contacts = self.contact_head(seq_tokens, attentions)
            #     result["contacts"] = contacts

        return result
