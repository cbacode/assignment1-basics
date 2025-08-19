from jaxtyping import Float, Int
import torch
from torch import Tensor, nn
from cs336_basics.transformer.transformer import TransformerBlock
from cs336_basics.transformer.embedding import Embedding
from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.linear import Linear

# uv run pytest -k test_transformer_lm
def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """
    Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    t = Transformer(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    # with open("test.txt", "w") as f:
    #     print(weights, file = f)
    t.load_state_dict(weights)
    return t.forward(in_indices)

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        
        self.embedding = Embedding(vocab_size, d_model, self.device, self.dtype)
        self.attn_layers = nn.ModuleList(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, self.device, self.dtype) for _ in range(self.num_layers))
        self.rms_norm = RMSNorm(d_model, device = self.device)
        self.lm_head = Linear(d_model, vocab_size, self.device, self.dtype)
        
    def load_state_dict(self, weights: dict[str, Tensor]):
        """ 
        embedding: `token_embeddings.weight`
        attns: `layers.{num_layers}.attn.q_proj.weight`, ..., `layers.{num_layers}.ln1.weight`, `layers.{num_layers}.ffn.w1.weight`, ..., `layers.{num_layers}.ln2.weight`
        rms_norm: `ln_final.weight`
        lm_head: `lm_head.weight`
        """
        self.embedding.load_state_dict({"w": weights["token_embeddings.weight"]})
        for i in range(self.num_layers):
            self.attn_layers._modules[str(i)].load_state_dict({
                "attn.q_proj.weight": weights[f"layers.{str(i)}.attn.q_proj.weight"],
                "attn.k_proj.weight": weights[f"layers.{str(i)}.attn.k_proj.weight"],
                "attn.v_proj.weight": weights[f"layers.{str(i)}.attn.v_proj.weight"],
                "attn.output_proj.weight": weights[f"layers.{str(i)}.attn.output_proj.weight"],
                "ln1.weight": weights[f"layers.{str(i)}.ln1.weight"],
                "ffn.w1.weight": weights[f"layers.{str(i)}.ffn.w1.weight"],
                "ffn.w2.weight": weights[f"layers.{str(i)}.ffn.w2.weight"],
                "ffn.w3.weight": weights[f"layers.{str(i)}.ffn.w3.weight"],
                "ln2.weight": weights[f"layers.{str(i)}.ln2.weight"]
            })
        self.rms_norm.load_state_dict({"g": weights["ln_final.weight"]})
        self.lm_head.load_state_dict({"w": weights["lm_head.weight"]})
        
    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.embedding(in_indices)
        for layer in self.attn_layers:
            x = layer(x)
        res = self.lm_head(self.rms_norm(x))
        return res
        