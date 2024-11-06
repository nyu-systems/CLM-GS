import torch
from torch import nn
from optimizer import UnifiedAdam
import cpu_adam
import numpy as np

N = 1

xyz = nn.Parameter(torch.ones((N, 3), dtype=torch.float32, device="cuda").requires_grad_(True))
opacity = nn.Parameter(torch.ones((N, 1), dtype=torch.float32, device="cuda").requires_grad_(True))
scaling = nn.Parameter(torch.ones((N, 3), dtype=torch.float32, device="cuda").requires_grad_(True))
rotation = nn.Parameter(torch.ones((N, 4), dtype=torch.float32, device="cuda").requires_grad_(True))
parameters = nn.Parameter(torch.ones((N, 48), dtype=torch.float32, pin_memory=True).requires_grad_(True))

l = [
    {
        "params": [xyz],
        "lr": 1e-1,
        "name": "xyz",
    },
    {
        "params": [opacity],
        "lr": 1e-2,
        "name": "opacity",
    },
    {
        "params": [scaling],
        "lr": 1e-3,
        "name": "scaling",
    },
    {
        "params": [rotation],
        "lr": 1e-4,
        "name": "rotation",
    },
    {
        "params": [parameters], # nn.Parameters
        "lr": 0.3, #NOTE: set this as lr of the first feature in concated tensor
        "name": "parameters"
    },
]

column_sizes = [3, 45]
column_lrs = [0.3, 0.6]

optimizer = UnifiedAdam(l, column_sizes, column_lrs)

xyz.grad = torch.ones_like(xyz)
opacity.grad = torch.ones_like(opacity)
scaling.grad = torch.ones_like(scaling)
rotation.grad = torch.ones_like(rotation)
parameters.grad = torch.ones_like(parameters)

parameters_back = nn.Parameter(torch.ones((N, 48), dtype=torch.float32, pin_memory=True).requires_grad_(True))
l_cpu = [
    {
        "params": [parameters_back], # nn.Parameters
        "lr": 0.3, #NOTE: set this as lr of the first feature in concated tensor
        "name": "parameters"
    },
]

optimizer_back = cpu_adam.FusedCPUAdam(
    l_cpu,
    columns_sizes=column_sizes,
    columns_lr=column_lrs,
    lr=0.0,
    bias_correction=True, # This True is required. 
    betas=(0.9, 0.999),
    eps=1e-15,
    weight_decay=0,
    amsgrad=False,
    adamw_mode=False,
    fp32_optimizer_states=True
)

parameters_back.grad = torch.ones_like(parameters_back)

for param_group in optimizer.param_groups:
    bsz = 16
    lr_scale = np.sqrt(bsz)
    param_group["lr"] *= lr_scale
    
    if "eps" in param_group:  # Adam
        param_group["eps"] /= lr_scale
        param_group["betas"] = [beta**bsz for beta in param_group["betas"]]
optimizer.columns_lr *= lr_scale

for param_group in optimizer_back.param_groups:
    bsz = 16
    lr_scale = np.sqrt(bsz)
    param_group["lr"] *= lr_scale
    if "eps" in param_group:  # Adam
        param_group["eps"] /= lr_scale
        param_group["betas"] = [beta**bsz for beta in param_group["betas"]]
optimizer_back.columns_lr *= lr_scale



for i in range(10):
    optimizer.step()
    optimizer_back.step()

print("param_groups", optimizer.param_groups)
print("param_groups_back", optimizer_back.param_groups)


# xyz 
# opacity = nn.Parameter(torch.ones((N, 1), dtype=torch.float32, device="cuda").requires_grad_(True))
# scaling = nn.Parameter(torch.ones((N, 3), dtype=torch.float32, device="cuda").requires_grad_(True))
# rotation = nn.Parameter(torch.ones((N, 4), dtype=torch.float32, device="cuda").requires_grad_(True))
# parameters