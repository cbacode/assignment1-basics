from jaxtyping import Float
import torch
from torch import Tensor

# uv run pytest -k test_softmax_matches_pytorch

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_item = torch.max(in_features, dim = dim, keepdim = True).values
    for_add = torch.exp(in_features - max_item)
    add = torch.sum(for_add, dim = dim, keepdim = True)
    return for_add / add