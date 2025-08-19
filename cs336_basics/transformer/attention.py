from jaxtyping import Float, Bool, Int
import torch
from torch import Tensor, nn
from einops import einsum, rearrange
from cs336_basics.transformer.softmax import run_softmax
from cs336_basics.transformer.rope import RoPE
import math

# uv run pytest -k test_scaled_dot_product_attention
# uv run pytest -k test_4d_scaled_dot_product_attention
# https://www.runoob.com/pytorch/pytorch-transformer-model.html

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    
    """
    The hint here in the matrix is bad for understanding what is going on.
    The correct hint should like
        Q: Float[Tensor, " ... num_queries query"],
        K: Float[Tensor, " ... num_keys key"],
        V: Float[Tensor, " ... num_values d_v"],
        mask: Bool[Tensor, " ... num_queries num_keys"] | None = None,
    ) -> Float[Tensor, " ... num_queries(sequence_length) d_v"]:
    (sequence_length == num_queries == num_keys == num_values)
    
    Then QK^T means we multiple query and key using dot product, so we get a Tensor like Float[Tensor, " ... num_queries num_keys"], and after softmax function, each number in this tensor means the weight of values should be added to produce a new vector.
    We divide it by sqrt(d_k) because if we assume that each number is in normal distribution of (0, 1), than the division will push the distribution of sum back into (0, 1), hence improve the numerical stability.
    Finally we use the weight to combine a new value for each query, just like the answer for the question, and we will concat them together and multiply W_O to push the dimension of each vector back to d_model
    """
    qk = einsum(Q, K, " ... queries d_k,  ... keys d_k ->  ... queries keys")
    d_k = Q.size(-1)
    attn_scores = qk / math.sqrt(float(d_k))
    if mask is not None:
        attn_scores = torch.where(mask, attn_scores, float("-inf"))
    attn_probs = run_softmax(attn_scores, dim = -1)
    
    # len(keys) == len(values)
    return einsum(attn_probs, V, " ... queries keys,  ... keys d_v ->  ... queries d_v")

# uv run pytest -k test_multihead_self_attention
def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    
    """ 
    q_proj_weight: Float[Tensor, " hd_k d_model"],
    k_proj_weight: Float[Tensor, " hd_k d_model"],
    v_proj_weight: Float[Tensor, " hd_v d_model"],
    o_proj_weight: Float[Tensor, " d_model hd_v"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    (d_in == d_model)
    d_k is actually num_heads * d_k in pdf, that means num_heads * d_k == d_in using the usual setting of d_k = d_model(d_in) / num_heads.
    The reason for this design is after multiply q_proj_weight, k_proj_weight, v_proj_weight with in_features, you can get Q, K, V matrixs for all heads respectively.
    """
    hd_k = q_proj_weight.size(-2)
    assert hd_k % num_heads == 0
    d_k = hd_k // num_heads
    a = Attention(d_model, num_heads, d_k, d_k, device = q_proj_weight.device, dtype = q_proj_weight.dtype)
    a.load_state_dict({"q_proj_weight": q_proj_weight, "k_proj_weight": k_proj_weight, "v_proj_weight": v_proj_weight, "o_proj_weight": o_proj_weight})
    return a.forward(in_features)

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    hd_k = q_proj_weight.size(-2)
    sequence_length = in_features.size(-2)
    assert hd_k % num_heads == 0
    d_k = hd_k // num_heads
    a = Attention(d_model, num_heads, d_k, d_k, max_seq_len, theta, q_proj_weight.device, q_proj_weight.dtype)
    a.load_state_dict({"q_proj_weight": q_proj_weight, "k_proj_weight": k_proj_weight, "v_proj_weight": v_proj_weight, "o_proj_weight": o_proj_weight})
    return a.forward_with_rope(in_features, token_positions)


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, max_seq_len: int | None = None, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.hd_k = d_k * num_heads
        self.hd_v = d_v * num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        mu_k = 0
        sigma_k = math.sqrt(2.0 / (d_model + self.hd_k))
        left_bound_k = -3.0 * sigma_k
        right_bound_k = 3.0 * sigma_k
        
        mu_v = 0
        sigma_v = math.sqrt(2.0 / (d_model + self.hd_v))
        left_bound_v = -3.0 * sigma_v
        right_bound_v = 3.0 * sigma_v
        
        self.device = device
        self.dtype = dtype
        
        # q_proj_weight: Float[Tensor, " hd_k d_model"],
        # k_proj_weight: Float[Tensor, " hd_k d_model"],
        # v_proj_weight: Float[Tensor, " hd_v d_model"],
        # o_proj_weight: Float[Tensor, " d_model hd_v"],
        self.q_proj_weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.hd_k, self.d_model, device = self.device), mu_k, sigma_k, left_bound_k, right_bound_k).to(self.dtype))
        self.k_proj_weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.hd_k, self.d_model, device = self.device), mu_k, sigma_k, left_bound_k, right_bound_k).to(self.dtype))
        self.v_proj_weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.hd_v, self.d_model, device = self.device), mu_v, sigma_v, left_bound_v, right_bound_v).to(self.dtype))
        self.o_proj_weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.d_model, self.hd_v, device = self.device), mu_v, sigma_v, left_bound_v, right_bound_v).to(self.dtype))
        return
    
    def pre(self, in_features: Float[Tensor, " ... sequence_length d_in"]) -> tuple[Float[Tensor, "... num_heads sequence_length d_k"], Float[Tensor, "... num_heads sequence_length d_k"], Float[Tensor, "... num_heads sequence_length d_v"], Bool[Tensor, "sequence_length sequence_length"]]:
        sequence_length = in_features.size(-2)
        qkv_proj_weight = torch.cat([self.q_proj_weight, self.k_proj_weight, self.v_proj_weight], dim = 0)
        # print(self.q_proj_weight.size(), qkv_proj_weight.size())
        qkv = einsum(qkv_proj_weight, in_features, " three_hd_k d_in, ... sequence_length d_in -> ... sequence_length three_hd_k")
        # print(qkv.size())
        t = torch.ones(sequence_length, sequence_length, device = self.device)
        # Don't forget to transpose.
        mask = torch.triu(t).to(torch.bool).transpose(1, 0)
        # with open("test.txt", "w") as f:
        #     print(mask, file = f)
        
        """
        # This is bad because not parellel.
        res = []
        for i in range(self.num_heads):
            Q = qkv[..., (i * self.d_k):((i + 1) * self.d_k)]
            K = qkv[..., (self.hd_k + i * self.d_k):(self.hd_k + (i + 1) * self.d_k)]
            V = qkv[..., (2 * self.hd_k + i * self.d_k):(2 * self.hd_k + (i + 1) * self.d_k)]
            res.append(run_scaled_dot_product_attention(Q, K, V, mask))
        """
        Qs = qkv[..., 0:self.hd_k]
        Ks = qkv[..., self.hd_k:(2 * self.hd_k)]
        Vs = qkv[..., (2 * self.hd_k):]
        Qs = rearrange(Qs, "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k", num_heads = self.num_heads)
        Ks = rearrange(Ks, "... sequence_length (num_heads d_k) -> ...  num_heads sequence_length d_k", num_heads = self.num_heads)
        Vs = rearrange(Vs, "... sequence_length (num_heads d_v) -> ... num_heads sequence_length d_v", num_heads = self.num_heads)
        return (Qs, Ks, Vs, mask)
        
    def forward(self, in_features: Float[Tensor, " ... sequence_length d_in"]) -> Float[Tensor, " ... sequence_length d_out"]:
        Qs, Ks, Vs, mask = self.pre(in_features)
        res = run_scaled_dot_product_attention(Qs, Ks, Vs, mask)
        multihead = rearrange(res, "... num_heads sequence_length d_v -> ...  sequence_length (num_heads d_v)")
        return einsum(multihead, self.o_proj_weight, " ... sequence_length d_v, d_model d_v ->  ... sequence_length d_model")
    
    def forward_with_rope(self, in_features: Float[Tensor, " ... sequence_length d_in"], token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> Float[Tensor, " ... sequence_length d_out"]:
        Qs, Ks, Vs, mask = self.pre(in_features)
        assert self.max_seq_len is not None
        assert self.theta is not None
        sequence_length = in_features.size(-2)
        if token_positions is None:
            token_positions = torch.tensor(range(sequence_length), dtype = self.dtype, device = self.device)
        self.rope = RoPE(self.theta, self.d_k, self.max_seq_len, self.device)
        Qs = self.rope.forward(Qs, token_positions)
        Ks = self.rope.forward(Ks, token_positions)
        res = run_scaled_dot_product_attention(Qs, Ks, Vs, mask)
        multihead = rearrange(res, "... num_heads sequence_length d_v -> ...  sequence_length (num_heads d_v)")
        return einsum(multihead, self.o_proj_weight, " ... sequence_length d_v, d_model d_v ->  ... sequence_length d_model")