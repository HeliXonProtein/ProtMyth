# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F
import einops

from protmyth.modules.esm.axial_transformer import AxialTransformerLayer
from protmyth.modules.embeddings.seq2node import LearnedPositionalEmbedding, SinusoidalPositionalEmbedding,
from protmyth.modules.losses.auxilary import BertHead, ContactPredictHead



class ESM(nn.Module):
    """
    ProteinBertModel integrates components for encoding protein sequences
    with a transformer architecture specifically suited for protein data.

    Attributes
    ----------
    args : argparse.Namespace
        Parsed arguments specifying the model configuration.
    alphabet_size : int
        Size of the token alphabet used in the model.
    padding_idx : int
        Index used for padding tokens.
    mask_idx : int
        Index used for mask tokens.
    cls_idx : int
        Index used for CLS tokens.
    eos_idx : int
        Index used for EOS tokens.
    prepend_bos : bool
        Flag indicating if BOS token should be prepended.
    append_eos : bool
        Flag indicating if EOS token should be appended.
    emb_layer_norm_before : bool
        Flag indicating if layer norm is applied before embedding.
    model_version : str
        Version of the model, identifying distinct configurations.
    """

    @classmethod
    def add_args(cls, parser: Any) -> None:
        """
        Add model-specific arguments to the parser.

        Parameters
        ----------
        parser : ArgumentParser
            Argument parser to which model arguments are added.
        """
        parser.add_argument("--num_layers", default=36, type=int, metavar="N", help="number of layers")
        parser.add_argument("--embed_dim", default=1280, type=int, metavar="N", help="embedding dimension")
        parser.add_argument("--logit_bias", action="store_true", help="whether to apply bias to logits")
        parser.add_argument("--ffn_embed_dim", default=5120, type=int, metavar="N", help="embedding dimension for FFN")
        parser.add_argument("--attention_heads", default=20, type=int, metavar="N", help="number of attention heads")

    def __init__(self, args: Any, alphabet: Any) -> None:
        """
        Initialize the ProteinBertModel with the given arguments and alphabet.

        Parameters
        ----------
        args : Any
            Arguments specifying model parameters.
        alphabet : Any
            Class including alphabet-size and token indices for padding, masking, etc.
        """
        super().__init__()
        self.args = args
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.emb_layer_norm_before = getattr(self.args, "emb_layer_norm_before", False)

        # Select initialization based on architecture
        if self.args.arch == "roberta_large":
            self.model_version = "ESM-1b"
            self._init_submodules_esm1b()
        else:
            self.model_version = "ESM-1"
            self._init_submodules_esm1()

    def _init_submodules_common(self) -> None:
        """
        Initialize common submodules for the transformer model.
        """
        self.embed_tokens = nn.Embedding(self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([
            TransformerLayer(self.args.embed_dim, self.args.ffn_embed_dim, self.args.attention_heads,
                             add_bias_kv=(self.model_version != "ESM-1b"),
                             use_esm1b_layer_norm=(self.model_version == "ESM-1b"))
            for _ in range(self.args.num_layers)
        ])
        self.contact_head = ContactPredictionHead(self.args.num_layers * self.args.attention_heads,
                                                  self.prepend_bos, self.append_eos, eos_idx=self.eos_idx)

    def _init_submodules_esm1b(self) -> None:
        """
        Initialize ESM-1b specific submodules.
        """
        self._init_submodules_common()
        self.embed_scale = 1
        self.embed_positions = LearnedPositionalEmbedding(self.args.max_positions, self.args.embed_dim,
                                                          self.padding_idx)
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        self.lm_head = RobertaLMHead(embed_dim=self.args.embed_dim, output_dim=self.alphabet_size,
                                     weight=self.embed_tokens.weight)
        if self.emb_layer_norm_before:
            self.emb_layer_norm_before = ESM1bLayerNorm(self.args.embed_dim)

    def _init_submodules_esm1(self) -> None:
        """
        Initialize ESM-1 specific submodules.
        """
        self._init_submodules_common()
        self.embed_scale = math.sqrt(self.args.embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(self.args.embed_dim, self.padding_idx)
        self.embed_out = nn.Parameter(torch.zeros((self.alphabet_size, self.args.embed_dim)))
        self.embed_out_bias = None
        if getattr(self.args, "final_bias", False):
            self.embed_out_bias = nn.Parameter(torch.zeros(self.alphabet_size))

    def forward(self,
                seq_tokens: torch.Tensor,
                repr_layers: List[int] = [],
                return_contacts: bool = False,
                ) -> Dict[str, Any]:
        """
        Forward pass to compute representations and optionally predict contacts.

        Parameters
        ----------
        seq_tokens : torch.Tensor
            Input sequence tokens.
        repr_layers : List[int], optional
            Layers from which to extract hidden representations.
        return_contacts : bool, optional
            Whether to return contact predictions.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing model outputs.
        """
        if return_contacts:
            need_head_weights = True
        else:
            need_head_weights = False

        assert seq_tokens.ndim == 2, "Tokens tensor must be 2-dimensional"
        padding_mask = seq_tokens.eq(self.padding_idx)
        bert_mask = seq_tokens.eq(self.mask_idx)

        seq_embed = self.embed_scale * self.embed_tokens(seq_tokens)
        seq_embed = seq_embed + self.embed_positions(seq_embed)

        if self.model_version == "ESM-1b" and self.emb_layer_norm_before:
            seq_embed = self.emb_layer_norm_before(seq_embed)
            seq_embed = seq_embed * (1 - padding_mask.unsqueeze(-1).type_as(seq_embed))

        repr_layers_set = set(repr_layers)
        hidden_embeds = {}
        if 0 in repr_layers_set:
            hidden_embeds[0] = seq_embed

        if need_head_weights:
            attn_weights = []

        seq_embed = einops.rearrange(seq_embed, 'b t e -> t b e')  # Transpose from (B, T, E) to (T, B, E)

        for layer_idx, layer in enumerate(self.layers):
            seq_embed, attn = layer(seq_embed, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights)
            if (layer_idx + 1) in repr_layers_set:
                hidden_embeds[layer_idx + 1] = einops.rearrange(seq_embed, 't b e -> b t e')
            if need_head_weights:
                attn_weights.append(
                    einops.rearrange(attn, 'h b t1 t2 -> b h t1 t2'))  # Transpose from (H, B, T, T) to (B, H, T, T)

        if self.model_version == "ESM-1b":
            seq_embed = self.emb_layer_norm_after(seq_embed)
            seq_embed = einops.rearrange(seq_embed, 't b e -> b t e')  # Transpose back from (T, B, E) to (B, T, E)
            if (layer_idx + 1) in repr_layers_set:
                hidden_embeds[layer_idx + 1] = seq_embed
            seq_embed = self.lm_head(seq_embed)
        else:
            seq_embed = F.linear(seq_embed, self.embed_out, bias=self.embed_out_bias)
            seq_embed = einops.rearrange(seq_embed, 't b e -> b t e')  # Transpose back from (T, B, E) to (B, T, E)

        result = {"logits": seq_embed, "representations": hidden_representations}


        return result
