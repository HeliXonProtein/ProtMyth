import torch
from torch import nn
from jaxtyping import Float
import einops

from protmyth.modules.base import BaseModule
from protmyth.modules.register import register_module


@register_module("auxilary")
class BertHead(BaseModule[Float[torch.Tensor, "..."]]):
    """  
    A neural network module for processing multiple sequence alignments (MSA).  

    Parameters  
    ----------  
    d_msa : int  
        The dimensionality of the MSA input features.  
    d_out : int  
        The dimensionality of the output logits.  

    Methods  
    -------  
    forward(msa: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:  
        Computes the logits from the input MSA tensor.  
    """  

    def __init__(self, d_msa: int, d_out: int) -> None:  
        super().__init__()  
        self.linear_logits = nn.Linear(d_msa, d_out)  

    def forward(self,
                msa: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        """  
        Forward pass to compute logits from MSA input.  

        Parameters  
        ----------  
        msa : Float[torch.Tensor, "..."]  
            The input MSA tensor.  

        Returns  
        -------  
        Float[torch.Tensor, "..."]  
            The computed logits tensor.  
        """
        msa_rearranged = einops.rearrange(msa, 'b n d -> b (n d)')

        logits = self.linear_logits(msa_rearranged)  
        return logits
