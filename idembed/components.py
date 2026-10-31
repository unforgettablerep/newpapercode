"""Per-variable components that can be combined with the identity embedding.

- ChannelNorm: LayerNorm with per-variable affine parameters (channel normalisation, Lee et al.,
  ICML 2025), written as the shared LayerNorm affine plus a per-variable offset that starts at
  zero. At initialisation the layer equals LayerNorm, and a variable can fall back to the
  shared affine (keep = 0).
- RecurrentCycle: learnable per-variable cycles of length W (residual cycle forecasting,
  CycleNet, Lin et al., NeurIPS 2024). The cycle is removed from the normalised input and added
  back to the forecast; it starts at zero, so at initialisation the model is unchanged. With
  `shared`, the cycle is a shared cycle plus per-variable offsets, and only the offsets are
  variable-specific (papers/theory.md in the newpaper repository, D1).

Both take `keep` (B, n) in [0, 1]: 0 switches the variable-specific part off for that sample
and variable (variable-specific dropout during training, the drift gate at test time).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelNorm(nn.Module):
    """Only the first n_vars tokens (the variables) get their own parameters; calendar tokens
    use the shared ones."""

    def __init__(self, n_vars, d_model, eps=1e-5):
        super().__init__()
        self.eps, self.d = eps, d_model
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.d_weight = nn.Parameter(torch.zeros(n_vars, d_model))
        self.d_bias = nn.Parameter(torch.zeros(n_vars, d_model))

    def forward(self, x, var_idx=None, keep=None):
        z = F.layer_norm(x, (self.d,), eps=self.eps)
        y = z * self.weight + self.bias
        dw, db = (self.d_weight, self.d_bias) if var_idx is None else (self.d_weight[var_idx],
                                                                      self.d_bias[var_idx])
        n = dw.shape[0]
        dw, db = dw.unsqueeze(0).to(x.dtype), db.unsqueeze(0).to(x.dtype)
        if keep is not None:
            dw, db = dw * keep[..., None], db * keep[..., None]
        return torch.cat([y[:, :n] + z[:, :n] * dw + db, y[:, n:]], 1)


class RecurrentCycle(nn.Module):
    def __init__(self, n_vars, cycle, shared=False):
        super().__init__()
        self.cycle = cycle
        self.data = nn.Parameter(torch.zeros(cycle, n_vars))
        self.shared = nn.Parameter(torch.zeros(cycle, 1)) if shared else None

    def index(self, phase, start, length):
        return (phase[:, None] + start + torch.arange(length, device=phase.device)) % self.cycle

    def forward(self, phase, start, length, var_idx=None):
        """phase: (B,) absolute index of each window's first step. Returns (B, length, n), the
        variable-specific cycle values at steps start .. start + length - 1 of each window."""
        q = self.data if var_idx is None else self.data[:, var_idx]
        return q[self.index(phase, start, length)]

    def shared_part(self, phase, start, length):
        """(B, length, 1) values of the shared cycle, or 0 without one."""
        return 0.0 if self.shared is None else self.shared[self.index(phase, start, length)]
