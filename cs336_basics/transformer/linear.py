from einops import einsum
from jaxtyping import Float
import torch
from torch import nn
from torch import Tensor
import math

# uv run pytest -k test_linear

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    l = Linear(d_in, d_out, weights.device, weights.dtype)
    l.load_state_dict({"w": weights})
    return l.forward(in_features)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """ 
        Construct a linear transformation module. This function should accept the following parameters:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        mu = 0
        sigma = math.sqrt(2.0 / (in_features + out_features))
        left_bound = -3.0 * sigma
        right_bound = 3.0 * sigma
        
        self.device = device
        self.dtype = dtype
        
        self.w = nn.Parameter(nn.init.trunc_normal_(torch.empty(out_features, in_features, device = self.device), mu, sigma, left_bound, right_bound).to(self.dtype))
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Apply the linear transformation to the input
        """
        return einsum(self.w, x, " d_out d_in, ... d_in -> ... d_out")