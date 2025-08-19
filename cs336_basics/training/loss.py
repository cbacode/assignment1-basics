from torch import Tensor
import torch
from jaxtyping import Float, Int

# uv run pytest -k test_cross_entropy

def log_softmax(in_features: Float[Tensor, " seq_len vocab_size"], targets: Int[Tensor, " seq_len"]):
    max_item = torch.max(in_features, dim = -1, keepdim = True).values
    sub = in_features - max_item
    for_add = torch.exp(sub)
    add = torch.sum(for_add, dim = -1, keepdim = False)
    # print(targets)
    # print(in_features)
    # print(in_features[targets.unsqueeze(0)])
    # return torch.log(add) - in_features[..., targets]
    # forgot to shift back again
    o_i = torch.gather(sub, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return torch.log(add) - o_i

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    """ 
    inputs: Float[Tensor, " ... batch_size seq_len vocab_size"], 
    targets: Int[Tensor, " ... batch_size seq_len"]
    -> Float[Tensor, "... batchsize"]:
    """
    s = inputs.size()
    # print(inputs, targets, log_softmax(inputs, targets))
    return torch.sum(log_softmax(inputs, targets)) / s[0]