from collections.abc import Callable, Iterable
import torch
import math
from torch.nn.utils.clip_grad import clip_grad_norm_, _get_total_norm

# uv run pytest -k test_gradient_clipping

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # clip_grad_norm_(parameters, max_l2_norm)
    # grads = [p.grad for p in parameters if p.grad is not None]
    # total_norm = _get_total_norm(grads)
    # print(total_norm)
    
    # This shows that norm of all parameters are computed together
    # for para in parameters:
    #     clip_grad_norm_(para, max_l2_norm)
    
    # for para in parameters:
    #     if para.grad is None:
    #         continue
    #     norm = math.sqrt(torch.sum(para.grad.data * para.grad.data))
    #     if norm >= max_l2_norm:
    #         para.grad.data *= (max_l2_norm / (norm + 1e-6))
    
    # I don't know why it should like this.
    norm = 0.0
    for para in parameters:
        if para.grad is None:
            continue
        norm += torch.sum(para.grad.data * para.grad.data)
    norm = math.sqrt(norm)
    if norm >= max_l2_norm:
        for para in parameters:
            if para.grad is None:
                continue
            else:
                para.grad.data *= (max_l2_norm / (norm + 1e-6))
    