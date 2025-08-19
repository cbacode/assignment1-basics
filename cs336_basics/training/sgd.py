from collections.abc import Callable, Iterable
from torch import Tensor
from typing import Optional, DefaultDict, Any
import torch
import math

# uv run ./cs336_basics/training/sgd.py
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state: DefaultDict[Tensor, Any] = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
    
""" 
21.89389419555664
21.026893615722656
20.436368942260742
19.967138290405273
19.569791793823242
19.221282958984375
...
10.599008560180664
10.555782318115234
10.512953758239746
10.470519065856934
10.428467750549316
"""
print("#" * 50)
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)
for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.

   
""" 
# faster
25.4507999420166
16.28851318359375
12.007196426391602
9.394349098205566
7.6094231605529785
...
0.01474498026072979
0.014149162918329239
0.01358034648001194
0.013037159107625484
0.012518312782049179
"""   
print("#" * 50)
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1e1)
for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.

""" 
even faster
26.433759689331055
26.433759689331055
4.53531551361084
0.10854030400514603
9.062195729670416e-17
...
0.0
0.0
0.0
0.0
0.0
"""
print("#" * 50)
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1e2)
for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.
  
""" 
# diverge
21.4461612701416
7742.06298828125
1337176.375
148746560.0
12048471040.0
...
inf
inf
inf
inf
inf
"""  
print("#" * 50)
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1e3)
for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.