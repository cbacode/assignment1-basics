from collections.abc import Callable, Iterable
from torch import Tensor
from typing import Optional, DefaultDict, Any
import torch
import math

# uv run pytest -k test_adamw
# TODO
def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    # return AdamW
    return torch.optim.AdamW

# uv run ./cs336_basics/training/sgd.py
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, weight_decay = 0.1, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta_1": betas[0], "beta_2": betas[1], "eps": eps, "lambda": weight_decay}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            eps = group["eps"]
            lam = group["lambda"]
            # print(lr, beta_1, beta_2, eps, lam)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Get state associated with p.
                state = self.state[p]
                # Get iteration number from the state, or initial value.
                t = state.get("t", int(1))
                grad = p.grad.data
                m = state.get("m", torch.zeros(grad.size()))
                v = state.get("v", torch.zeros(grad.size()))
                # Get the gradient of loss with respect to p.
                
                m = beta_1 * m + (1.0 - beta_1) * grad
                v = beta_2 * v + (1.0 - beta_2) * grad * grad
                lr_t = lr / (1.0 - pow(beta_1, t))
                lr_t = lr_t * math.sqrt(1.0 - pow(beta_2, t))
                # Update weight tensor
                p.data *= (1.0 - lr * lam)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                
                state["t"] = t + 1 # Increment iteration number.
        return loss
#  ACTUAL: array([[ 0.844594,  0.882026,  0.273672],
# E              [ 0.023981, -0.626332, -0.385756]], dtype=float32)