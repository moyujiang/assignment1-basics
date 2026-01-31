import torch
import math
from torch import nn, Tensor
from typing import Iterable

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                t = state.get('step', 0) + 1
                state['step'] = t

                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq

                denom = (exp_avg_sq.sqrt() + group['eps'])
                step_size = group['lr'] * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))

                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

        return loss

def get_lr_cosine_schedule(t: int, lr_max: float, lr_min: float, T_w: int, T_c: int) -> float:
    """
    Compute the learning rate at time step t using a cosine schedule with warmup.

    Args:
        t (float): Current time step.
        lr_max (float): Maximum learning rate.
        lr_min (float): Minimum learning rate.
        T_w (float): Number of warmup steps.
        T_c (float): Total number of steps for cosine decay.

    Returns:
        float: The learning rate at time step t.
    """
    if t < T_w:
        lr = lr_max * (t / T_w)
    elif t < T_c:
        cosine_decay = 0.5 * (1 + math.cos(torch.pi * (t - T_w) / (T_c - T_w)))
        lr = lr_min + (lr_max - lr_min) * cosine_decay
    else:
        lr = lr_min
    return lr


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): Iterable of model parameters with gradients.
        max_l2_norm (float): Maximum allowed l2 norm for the gradients.
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)