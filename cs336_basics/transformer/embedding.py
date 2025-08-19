from einops import rearrange
from jaxtyping import Float, Int
import torch
from torch import nn
from torch import Tensor, LongTensor

# uv run pytest -k test_linear

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embed = Embedding(vocab_size, d_model, weights.device, weights.dtype)
    embed.load_state_dict({"w": weights})
    return embed.forward(token_ids)

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """ 
        Construct an embedding module. This function should accept the following parameters:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        mu = 0
        sigma = 1
        left_bound = -3.0 * sigma
        right_bound = 3.0 * sigma
        
        self.device = device
        self.dtype = dtype
        
        # TODO : We may use LongTensor.
        self.w = nn.Parameter(nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, device = self.device), mu, sigma, left_bound, right_bound).to(self.dtype))
        return
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """ 
        Lookup the embedding vectors for the given token IDs
        """
        return self.w[token_ids]
        