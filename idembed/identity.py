"""Learnable variable identity embeddings for variate tokens.

A variate-token model (iTransformer, S-Mamba) turns each variable's look-back window into
one token h_i = W x_i. Two variables with similar windows get similar tokens, so the model
cannot tell them apart; an identity vector e_i added to h_i fixes that:

    h_i <- h_i + e_i ,  i = 1..N

Modes
- none    : no identity (the original models).
- add     : a free table E in R^{N x D}.
- lowrank : E = U V with U in R^{N x r}, V in R^{r x D}; fewer parameters, a regulariser for
            datasets with few variables.
Every mode starts at E = 0, so at initialisation the model equals the original one.

`p_drop` (identity dropout) zeroes a variable's identity with probability p per sample during
training, so the model also learns to forecast without it; unseen variables at test time get
the zero identity (see `train_var_ratio` in the training script).
"""
import torch
import torch.nn as nn


class VariableIdentity(nn.Module):
    def __init__(self, n_vars, d_model, mode="add", rank=8, p_drop=0.0):
        super().__init__()
        if mode not in ("none", "add", "lowrank"):
            raise ValueError(mode)
        self.mode, self.p_drop, self.n_vars = mode, p_drop, n_vars
        # variables whose identity is used; the unseen-variable protocol switches unseen ones off
        self.register_buffer("active", torch.ones(n_vars, dtype=torch.bool))
        if mode == "add":
            self.table = nn.Parameter(torch.zeros(n_vars, d_model))
        elif mode == "lowrank":
            # a private generator keeps the global RNG (and so every other initialisation)
            # identical to the model without identity embeddings
            g = torch.Generator().manual_seed(20240 + n_vars)
            self.U = nn.Parameter(torch.randn(n_vars, rank, generator=g) / rank ** 0.5)
            self.V = nn.Parameter(torch.zeros(rank, d_model))

    def weight(self):
        if self.mode == "add":
            return self.table
        if self.mode == "lowrank":
            return self.U @ self.V
        return None

    def forward(self, h, var_idx=None):
        """h: (B, n + M, D) tokens; the first n are variables (in the order of var_idx),
        the remaining M are calendar-feature tokens, which get no identity."""
        if self.mode == "none":
            return h
        E = self.weight() * self.active[:, None]
        E = E if var_idx is None else E[var_idx]
        n = E.shape[0]
        E = E.unsqueeze(0).to(h.dtype)
        if self.training and self.p_drop > 0:
            keep = torch.rand(h.shape[0], n, 1, device=h.device) >= self.p_drop
            E = E * keep.to(h.dtype)
        return torch.cat([h[:, :n] + E, h[:, n:]], 1)
