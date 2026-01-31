import torch
from torch import nn, Tensor

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        t = torch.arange(max_seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos_buffer = torch.cos(freqs)
        sin_buffer = torch.sin(freqs)
        self.register_buffer("cos_buffer", cos_buffer)
        self.register_buffer("sin_buffer", sin_buffer)
    
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos = self.cos_buffer[token_positions].to(dtype=x.dtype)
        sin = self.sin_buffer[token_positions].to(dtype=x.dtype)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot_even = x1 * cos - x2 * sin
        rot_odd = x1 * sin + x2 * cos
        x_rotated = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)

        return x_rotated