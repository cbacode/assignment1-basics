from collections.abc import Callable, Iterable
import torch
import math

# uv run pytest -k test_gradient_clipping
# TODO
def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    for para in parameters:
        if para.grad is None:
            continue
        norm = math.sqrt(torch.sum(para.grad.data * para.grad.data))
        print(norm, max_l2_norm)
        # norm = torch.norm(para.grad.data, p=2)
        if norm >= max_l2_norm:
            para.grad.data *= (max_l2_norm / (norm + 1e-6))