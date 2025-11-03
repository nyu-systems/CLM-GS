import torch
import cpu_adam
from clm_kernels import selective_adam_update

class SelectiveAdam(torch.optim.Adam):
    """
    A custom optimizer that extends the standard Adam optimizer by
    incorporating selective updates.

    This class is useful for situations where only a subset of parameters
    should be updated at each step, such as in sparse models or in cases where
    parameter visibility is controlled by an external mask.

    Additionally, the operations are fused into a single kernel. This optimizer
    leverages the `selective_adam_update` function from a CUDA backend for
    optimized sparse updates.

    This is one of the two optimizers mentioned in the Taming3DGS paper.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        eps (float): Term added to the denominator to improve numerical stability (default: 1e-8).
        betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).

    Examples:

        >>> N = 100
        >>> param = torch.randn(N, requires_grad=True)
        >>> optimizer = SelectiveAdam([param], eps=1e-8, betas=(0.9, 0.999))
        >>> visibility_mask = torch.cat([torch.ones(50), torch.zeros(50)])  # Visible first half, hidden second half

        >>> # Forward pass
        >>> loss = torch.sum(param ** 2)

        >>> # Backward pass
        >>> loss.backward()

        >>> # Optimization step with selective updates
        >>> optimizer.step(visibility=visibility_mask)

    """

    def __init__(self, params, eps, betas):
        super().__init__(params=params, eps=eps, betas=betas)

    @torch.no_grad()
    def step(self, visibility):
        N = visibility.numel()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )

            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]
            M = param.numel() // N

            selective_adam_update(
                param,
                param.grad,
                exp_avg,
                exp_avg_sq,
                visibility,
                lr,
                beta1,
                beta2,
                eps,
                N,
                M,
            )  

class UnifiedAdam(torch.optim.Optimizer):
    """Provide a unified interface for two independent Adam optimizers
    on both GPU and CPU.
    """
    def __init__(
        self,
        params,
        columns_sizes,
        columns_lr,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-15, #NOTE: Adam default eps is 1e-8. It's our intent to use 1e-15 in 3DGS.  
        weight_decay=0,
        amsgrad=False,
        adamw_mode=False,
        fp32_optimizer_states=True,
        fused=False,
        sparse=False,
    ):
        
        params_device = []
        params_host = []
        
        for p in params:
            if p["name"] == "parameters":
                assert p["params"][0].is_pinned()
                params_host.append(p)
            else:
                assert p["params"][0].is_cuda
                params_device.append(p)

        if sparse:
            self.gpu_adam = SelectiveAdam(params_device, eps=eps, betas=betas)
        else:
            self.gpu_adam = torch.optim.Adam(params_device, lr=0.0, eps=eps, fused=fused)
        self.cpu_adam = cpu_adam.FusedCPUAdam(
            params_host,
            columns_sizes=columns_sizes,
            columns_lr=columns_lr,
            lr=0.0,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            adamw_mode=adamw_mode,
            fp32_optimizer_states=fp32_optimizer_states
        )
        
        self.columns_lr = self.cpu_adam.columns_lr
        
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            adamw_mode=adamw_mode,
            fp32_optimizer_states=fp32_optimizer_states
        )
        
        # super(UnifiedAdam, self).__init__(params, defaults)  # -> not root cause of nan bug
        self.param_groups = self.gpu_adam.param_groups + self.cpu_adam.param_groups
        self.state = self.gpu_adam.state | self.cpu_adam.state #NOTE: This works but is weird: optimizer.states will be on both host & device
        
        # self.param_groups = self.gpu_adam.param_groups + self.cpu_adam.param_groups
    
        
    # def __setstate__(self, state):
    #     super(UnifiedAdam, self).__setstate__(state)     
    
    # def _init_group(self):
    #     pass
    
    # def __del__(self):
    #     self.cpu_adam.__del__()
    
    def get_all_states(self):
        return [self.gpu_adam.state, self.cpu_adam.state]
    
    def zero_grad(self, set_to_none=False):
        self.gpu_adam.zero_grad(set_to_none)
        self.cpu_adam.zero_grad(set_to_none)
    
    def step(self, closure=None):
        self.gpu_adam.step()
        self.cpu_adam.step()
        self.state = self.gpu_adam.state | self.cpu_adam.state
        
        