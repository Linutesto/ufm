from __future__ import annotations

import torch


class OffloadedAdam(torch.optim.Optimizer):
    """
    Minimal AdamW-like optimizer with CPU-offloaded master params + states.

    - Keeps FP32 master params and optimizer states on CPU (pinned memory)
    - Applies updates on CPU, then copies results back to the model tensor
      device/dtype
    - Intended for single-GPU offload; not a drop-in ZeRO replacement, but
      effective for memory relief (removes the 2-3x optimizer-state VRAM cost).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                mp = p.detach().float().cpu().pin_memory().clone()
                state["master_param"] = mp
                state["exp_avg"] = torch.zeros_like(mp, memory_format=torch.preserve_format).pin_memory()
                state["exp_avg_sq"] = torch.zeros_like(mp, memory_format=torch.preserve_format).pin_memory()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]; beta1, beta2 = group["betas"]; eps = group["eps"]; wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                g = p.grad.detach()
                grad_cpu = g.float().cpu()
                mp = state["master_param"]; m = state["exp_avg"]; v = state["exp_avg_sq"]
                if wd != 0:
                    grad_cpu.add_(mp, alpha=wd)
                m.mul_(beta1).add_(grad_cpu, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1 - beta2)
                denom = v.sqrt().add_(eps)
                step = m / denom
                mp.add_(step, alpha=-lr)
                # Copy updates back to model tensor device/dtype
                p.copy_(mp.to(device=p.device, non_blocking=True).to(dtype=p.dtype))
        return loss
