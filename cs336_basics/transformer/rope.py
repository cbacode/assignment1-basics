from einops import einsum, rearrange
from jaxtyping import Float, Int
import torch
from torch import nn
from torch import Tensor
import math

# uv run pytest -k test_rope

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    l = RoPE(theta, d_k, max_seq_len, in_query_or_key.device)
    return l.forward(in_query_or_key, token_positions)

def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """ 
        Construct the RoPE module and create buffers if needed.
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """
        if not hasattr(self, 'cache'):
            super().__init__()
            self.theta = theta
            self.d_k = d_k
            self.max_seq_len = max_seq_len
            self.device = device
            # Compute and save cache
            cache = []
            for i in range(max_seq_len):
                inside_cache = []
                for k in range(d_k // 2):
                    exp = float(2 * k) / float(d_k)
                    angle = float(i) / pow(theta, exp)
                    arr = torch.tensor([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]], device = device)
                    inside_cache.append(arr)
                cache.append(torch.stack(inside_cache))
            cache = torch.stack(cache)
            
            self.register_buffer("cache", cache, persistent=False)
        return
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """ 
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note that you should tolerate x with an arbitrary number of batch dimensions. You should assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors along the sequence dimension
        """
        # rot is a (..., seq_len, d_k / 2, 2, 2) Tensor
        # if rot is a (..., seq_len, d_k, d_k) Tensor, then we can directly multiply it.
        rot = self.cache[token_positions]
        
        x = rearrange(x, " ... seq_len (d_k_2 two) -> ... seq_len d_k_2 two", two = 2)
        res = einsum(x, rot, "... seq_len d_k_2 col, ... seq_len d_k_2 col row -> ... seq_len d_k_2 row")
        return rearrange(res, "... seq_len d_k_2 row -> ... seq_len (d_k_2 row)")
    
    def brute_forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        d_k = x.size(-1)
        # Compute and save cache
        cache = []
        for i in range(self.max_seq_len):
            inside_cache = torch.tensor([[0.0] * d_k] * d_k)
            for k in range(d_k // 2):
                exp = float(2 * k) / float(d_k)
                angle = float(i) / pow(self.theta, exp)
                inside_cache[2 * k][2 * k] = math.cos(angle)
                inside_cache[2 * k][2 * k + 1] = math.sin(angle)
                inside_cache[2 * k + 1][2 * k] = -math.sin(angle)
                inside_cache[2 * k + 1][2 * k + 1] = math.cos(angle)
            cache.append(inside_cache)
        cache = torch.stack(cache)
        
        rot = cache[token_positions]
        return einsum(x, rot, "... seq_len d_k_1, ... seq_len d_k_1 d_k -> ... seq_len d_k")