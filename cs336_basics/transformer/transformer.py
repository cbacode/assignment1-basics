from jaxtyping import Float
import torch
from torch import Tensor, nn
from cs336_basics.transformer.attention import Attention
from cs336_basics.transformer.swiglu import SwiGLU
from cs336_basics.transformer.rmsnorm import RMSNorm

# uv run pytest -k test_transformer_block
def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    t = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)
    t.load_state_dict(weights)
    return t.forward(in_features)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        
        d_k = d_model // num_heads
        self.attn = Attention(d_model, num_heads, d_k, d_k, max_seq_len, theta, self.device, self.dtype)
        self.norm_attn = RMSNorm(d_model, device = self.device)
        self.ffn = SwiGLU(d_model, d_ff, self.device, self.dtype)
        self.norm_ffn = RMSNorm(d_model, device = self.device)
        return
    
    def load_state_dict(self, weights: dict[str, Tensor]):
        """ 
        attn: `attn.q_proj.weight`, `attn.k_proj.weight`, `attn.v_proj.weight`, `attn.output_proj.weight`
        attn_norm: `ln1.weight`
        ffn: `ffn.w1.weight`, `ffn.w2.weight`, `ffn.w3.weight`
        ffn_norm: `ln2.weight`
        """
        self.attn.load_state_dict({"q_proj_weight": weights["attn.q_proj.weight"], "k_proj_weight":weights["attn.k_proj.weight"], "v_proj_weight":weights["attn.v_proj.weight"], "o_proj_weight":weights["attn.output_proj.weight"]})
        self.norm_attn.load_state_dict({"g": weights["ln1.weight"]})
        self.ffn.load_state_dict({"w1": weights["ffn.w1.weight"], "w2": weights["ffn.w2.weight"], "w3": weights["ffn.w3.weight"]})
        self.norm_ffn.load_state_dict({"g": weights["ln2.weight"]})
        return
    
    def forward(self, in_features: Float[Tensor, " batch sequence_length d_model"]) -> Float[Tensor, " batch sequence_length d_model"]:
        attn = in_features + self.attn.forward_with_rope(self.norm_attn.forward(in_features))
        # attn = in_features + self.attn.forward(self.norm_attn.forward(in_features))
        res = attn + self.ffn.forward(self.norm_ffn.forward(attn))
        # attn = self.attn.forward_with_rope(self.norm_attn.forward(in_features))
        # res = in_features + self.ffn.forward(self.norm_ffn.forward(attn))
        return res