from einops import einsum
from jaxtyping import Float
import torch
from torch import nn
from torch import Tensor
import math

# uv run pytest -k test_swiglu

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
        
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    """
    l = SwiGLU(d_model, d_ff, w1_weight.device, w1_weight.dtype)
    l.load_state_dict({"w1": w1_weight, "w2": w2_weight, "w3": w3_weight})
    return l.forward(in_features)

class SwiGLU(nn.Module):
    d_model: int
    d_ff: int
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """ 
        Construct a swiglu transformation module. This function should accept the following parameters:
            d_model: int final dimension of the input
            d_ff: int final dimension inside
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.mu = 0
        self.sigma = math.sqrt(2.0 / (d_model + d_ff))
        self.left_bound = -3.0 * self.sigma
        self.right_bound = 3.0 * self.sigma
        
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w1 = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.d_ff, self.d_model, device = self.device), self.mu, self.sigma, self.left_bound, self.right_bound).to(self.dtype))
        self.w2 = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.d_model, self.d_ff, device = self.device), self.mu, self.sigma, self.left_bound, self.right_bound).to(self.dtype))
        self.w3 = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.d_ff, self.d_model, device = self.device), self.mu, self.sigma, self.left_bound, self.right_bound).to(self.dtype))
        return
    
    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Apply the swiglu transformation to the input
        """
        arr2 = self.silu(einsum(self.w1, x, " d_ff d_model, ... d_model -> ... d_ff"))
        arr3 = einsum(self.w3, x, " d_ff d_model, ... d_model -> ... d_ff")
        return einsum(self.w2, arr2 * arr3, " d_model d_ff, ... d_ff -> ... d_model")