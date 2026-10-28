"""Mamba block: the official CUDA implementation (`mamba_ssm`) when available on a GPU,
otherwise a pure-PyTorch implementation with the same parameterisation (slower, any device)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaTorch(nn.Module):
    """Mamba (S6) block in plain PyTorch, parameterised like `mamba_ssm.Mamba`.

    The selective scan h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t is evaluated chunk-wise in log
    space (h_t = e^{S_t} cumsum_s e^{-S_s} u_s inside a chunk) and the chunks are chained
    through the carried state. It runs in float32 even under autocast; per-step log-decays are
    clamped at -10 (a factor below 5e-5) so that e^{-S} stays finite.
    """

    def __init__(self, d_model, d_state=16, d_conv=2, expand=1, chunk=4):
        super().__init__()
        d_inner = expand * d_model
        self.d_state, self.chunk, self.dt_rank = d_state, chunk, math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        dt = dt.clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))      # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def scan(self, dA, u, C):                         # (b, L, d, s), (b, L, d, s), (b, L, s)
        b, L, d, s = dA.shape
        l = self.chunk
        pad = (-L) % l
        if pad:                                        # zero input and no decay
            dA, u = F.pad(dA, (0, 0, 0, 0, 0, pad)), F.pad(u, (0, 0, 0, 0, 0, pad))
            C = F.pad(C, (0, 0, 0, pad))
        nc = (L + pad) // l
        S = torch.cumsum(dA.clamp(min=-10.0).view(b, nc, l, d, s), 2)
        local = torch.exp(S) * torch.cumsum(torch.exp(-S) * u.view(b, nc, l, d, s), 2)
        h, carry = dA.new_zeros(b, d, s), []
        for c in range(nc):
            carry.append(h)
            h = local[:, c, -1] + torch.exp(S[:, c, -1]) * h
        hs = local + torch.exp(S) * torch.stack(carry, 1)[:, :, None]
        y = torch.einsum("bclds,bcls->bcld", hs, C.view(b, nc, l, s))
        return y.reshape(b, nc * l, d)[:, :L]

    def forward(self, x):                              # (b, L, d_model)
        L = x.shape[1]
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            u, z = self.in_proj(x).chunk(2, -1)
            u = F.silu(self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2))
            dt, B, C = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], -1)
            dt = F.softplus(self.dt_proj(dt))
            A = -torch.exp(self.A_log)
            y = self.scan(dt[..., None] * A, dt[..., None] * B[:, :, None] * u[..., None], C)
            return self.out_proj((y + u * self.D) * F.silu(z))


def cuda_mamba_available():
    if not torch.cuda.is_available():
        return False
    try:
        import mamba_ssm  # noqa: F401
        return True
    except ImportError:
        return False


def build_mamba(d_model, d_state, d_conv=2, expand=1, impl="auto"):
    """impl: 'cuda' (mamba_ssm, required), 'torch' (pure PyTorch) or 'auto'."""
    if impl == "cuda" or (impl == "auto" and cuda_mamba_available()):
        from mamba_ssm import Mamba
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    if impl not in ("auto", "torch"):
        raise ValueError(impl)
    return MambaTorch(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
