from einops import einsum
from jaxtyping import Float
import torch
from torch import nn
from torch import Tensor
import math

# uv run pytest -k test_silu

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    silu = SiLU(in_features.device, in_features.dtype)
    return silu.forward(in_features)

class SiLU(nn.Module):
    def __init__(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """ 
        Construct a silu transformation module. This function should accept the following parameters:
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        return
    
    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Apply the swiglu transformation to the input
        """
        return x * self.sigmoid(x)