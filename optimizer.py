import torch
import cpu_adam

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
        fp32_optimizer_states=True):
        
        params_device = []
        params_host = []
        
        for p in params:
            if p["name"] == "parameters":
                assert p["params"][0].is_pinned()
                params_host.append(p)
            else:
                assert p["params"][0].is_cuda
                params_device.append(p)
            
        self.gpu_adam = torch.optim.Adam(params_device, lr=0.0, eps=eps)
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
        
        super(UnifiedAdam, self).__init__(params, defaults) 
        
        # self.param_groups = self.gpu_adam.param_groups + self.cpu_adam.param_groups
    
        
    # def __setstate__(self, state):
    #     super(UnifiedAdam, self).__setstate__(state)     
    
    # def _init_group(self):
    #     pass
    
    # def __del__(self):
    #     self.cpu_adam.__del__()
    
    def zero_grad(self, set_to_none=False):
        self.gpu_adam.zero_grad(set_to_none)
        self.cpu_adam.zero_grad(set_to_none)
    
    def step(self, closure=None):
        self.gpu_adam.step()
        self.cpu_adam.step()
        
        