from einops import einsum
from jaxtyping import Float
import torch
from torch import nn
from torch import Tensor
import math

# uv run pytest -k test_rmsnorm
def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    l = RMSNorm(d_model, eps, weights.device)
    l.load_state_dict({"g": weights})
    return l.forward(in_features)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None):
        """ 
        Construct the RMSNorm module. This function should accept the following parameters:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        mu = 0
        sigma = 1
        left_bound = -2
        right_bound = 2
        
        self.d_model = d_model
        self.eps = eps
        self.device = device
        
        self.g = nn.Parameter(nn.init.trunc_normal_(torch.empty(d_model, device = self.device), mu, sigma, left_bound, right_bound).to(torch.float32))
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        res_dtype = x.dtype
        x = x.to(torch.float32)
        sum_x_square = torch.sum(torch.square(x), -1)
        coff = torch.sqrt(sum_x_square / float(self.d_model) + self.eps)
        res = (x / coff.unsqueeze(-1)) * self.g
        return res.to(res_dtype)
    