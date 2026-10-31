"""iTransformer and S-Mamba (variate-token forecasters) with variable identity embeddings.

Both follow the official code bases: series-wise instance normalisation, one token per
variable (plus one per calendar feature) from Linear(seq_len -> d_model), e_layers encoder
layers, a final LayerNorm and Linear(d_model -> pred_len). The only change is the optional
identity embedding added to the variable tokens, either once after the embedding ('input')
or before every encoder layer ('every').

Optional per-variable components (idembed/components.py): channel normalisation in every
encoder LayerNorm (`cn`) and learnable per-variable cycles (`rcf_cycle` = W > 0). With
`vs_dropout` = p, all variable-specific parts (identity, channel-norm offsets, cycles) of a
variable are switched off together with probability p per sample during training, so that
the model can also forecast a variable without them; `keep` in forward() does the same at
test time (the drift gate).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import ChannelNorm, RecurrentCycle
from .identity import VariableIdentity
from .mamba import MambaTorch, build_mamba


def make_norm(d_model, n_vars, cn):
    return ChannelNorm(n_vars, d_model) if cn else nn.LayerNorm(d_model)


def apply_norm(norm, x, var_idx=None, keep=None):
    return norm(x, var_idx, keep) if isinstance(norm, ChannelNorm) else norm(x)


class DataEmbeddingInverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):                     # (B, L, N), (B, L, M) or None
        x = x.permute(0, 2, 1)
        if x_mark is not None:
            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)
        return self.dropout(self.value_embedding(x))  # (B, N + M, D)


class FullAttention(nn.Module):
    """Multi-head self-attention over tokens (AttentionLayer + FullAttention, no mask)."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        d_k = d_model // n_heads
        self.h, self.p = n_heads, dropout
        self.q, self.k, self.v = (nn.Linear(d_model, d_k * n_heads) for _ in range(3))
        self.o = nn.Linear(d_k * n_heads, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = (f(x).view(B, T, self.h, -1).transpose(1, 2) for f in (self.q, self.k, self.v))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv1d(d_model, d_ff, 1), nn.Conv1d(d_ff, d_model, 1)
        self.act = F.gelu if activation == "gelu" else F.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, y):
        y = self.dropout(self.act(self.conv1(y.transpose(-1, 1))))
        return self.dropout(self.conv2(y).transpose(-1, 1))


class TransformerLayer(nn.Module):
    """iTransformer EncoderLayer: x = LN(x + Drop(Attn(x))); x = LN(x + FFN(x))."""

    def __init__(self, d_model, n_heads, d_ff, dropout, activation, n_vars=0, cn=False):
        super().__init__()
        self.attn = FullAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1, self.norm2 = make_norm(d_model, n_vars, cn), make_norm(d_model, n_vars, cn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, var_idx=None, keep=None):
        x = apply_norm(self.norm1, x + self.dropout(self.attn(x)), var_idx, keep)
        return apply_norm(self.norm2, x + self.ffn(x), var_idx, keep)


class MambaLayer(nn.Module):
    """S-Mamba EncoderLayer: a forward and a reversed Mamba over the token sequence, summed;
    x = LN(x + Mamba(x) + flip(Mamba_r(flip(x)))); x = LN(x + FFN(x)).

    scan_order='shuffle' scans the tokens in a fresh random order at every training step, so
    a token's position in the scan carries no information about which variable it is. Setting
    `eval_orders` = K > 1 (evaluation only) averages the mixing output over K fixed random
    orders; K = 1 scans in the original order."""

    def __init__(self, d_model, d_state, d_ff, dropout, activation, mamba_impl="auto",
                 scan_order="fixed", n_vars=0, cn=False):
        super().__init__()
        if scan_order not in ("fixed", "shuffle"):
            raise ValueError(scan_order)
        self.scan_order, self.eval_orders, self.eval_seed = scan_order, 1, None
        self.fwd = build_mamba(d_model, d_state, d_conv=2, expand=1, impl=mamba_impl)
        self.bwd = build_mamba(d_model, d_state, d_conv=2, expand=1, impl=mamba_impl)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1, self.norm2 = make_norm(d_model, n_vars, cn), make_norm(d_model, n_vars, cn)

    def mix(self, x, perm=None):
        if perm is None:
            return self.fwd(x) + self.bwd(x.flip(1)).flip(1)
        xp = x[:, perm]
        return (self.fwd(xp) + self.bwd(xp.flip(1)).flip(1))[:, torch.argsort(perm)]

    def forward(self, x, var_idx=None, keep=None):
        T = x.shape[1]
        if self.training and self.scan_order == "shuffle":
            x = apply_norm(self.norm1, x + self.mix(x, torch.randperm(T, device=x.device)),
                           var_idx, keep)
        elif not self.training and self.eval_orders > 1:
            g = torch.Generator().manual_seed(12345)
            m = torch.stack([self.mix(x, torch.randperm(T, generator=g).to(x.device))
                             for _ in range(self.eval_orders)]).mean(0)
            x = apply_norm(self.norm1, x + m, var_idx, keep)
        elif not self.training and self.eval_seed is not None:      # one random order
            perm = torch.randperm(T, generator=torch.Generator().manual_seed(self.eval_seed))
            x = apply_norm(self.norm1, x + self.mix(x, perm.to(x.device)), var_idx, keep)
        else:                                       # original order, same arithmetic as S-Mamba
            x = apply_norm(self.norm1, x + self.fwd(x) + self.bwd(x.flip(1)).flip(1), var_idx, keep)
        return apply_norm(self.norm2, x + self.ffn(x), var_idx, keep)


class VariateForecaster(nn.Module):
    def __init__(self, backbone, n_vars, seq_len, pred_len, d_model=512, d_ff=512, e_layers=2,
                 n_heads=8, d_state=16, dropout=0.1, activation="gelu", use_norm=True,
                 identity="none", id_rank=8, id_loc="input", id_dropout=0.0, mamba_impl="auto",
                 scan_order="fixed", cn=False, rcf_cycle=0, vs_dropout=0.0, rcf_shared=False):
        super().__init__()
        self.use_norm, self.pred_len, self.id_loc = use_norm, pred_len, id_loc
        self.seq_len, self.vs_dropout = seq_len, vs_dropout
        self.embedding = DataEmbeddingInverted(seq_len, d_model, dropout)
        if backbone == "itransformer":
            if scan_order != "fixed":
                raise ValueError("scan_order applies to S-Mamba only")
            make = lambda: TransformerLayer(d_model, n_heads, d_ff, dropout, activation, n_vars, cn)
        elif backbone == "smamba":
            make = lambda: MambaLayer(d_model, d_state, d_ff, dropout, activation, mamba_impl,
                                      scan_order, n_vars, cn)
        else:
            raise ValueError(backbone)
        self.layers = nn.ModuleList(make() for _ in range(e_layers))
        self.norm = make_norm(d_model, n_vars, cn)
        self.projector = nn.Linear(d_model, pred_len)
        n_id = 1 if id_loc == "input" else e_layers
        if id_loc not in ("input", "every"):
            raise ValueError(id_loc)
        # created last so that the other layers get exactly the original initialisation
        self.identity = nn.ModuleList(VariableIdentity(n_vars, d_model, identity, id_rank, id_dropout)
                                      for _ in range(n_id))
        self.cycle = RecurrentCycle(n_vars, rcf_cycle, rcf_shared) if rcf_cycle else None

    def forward(self, x, x_mark=None, var_idx=None, phase=None, keep=None):
        """x: (B, L, n) with the variables listed in var_idx (default: all, in order).
        phase: (B,) absolute index of each window's first step (needed with rcf_cycle).
        keep: (B, n) in [0, 1], scales the variable-specific parts (1 = use them)."""
        n = x.shape[-1]
        if self.training and self.vs_dropout > 0:
            drop = (torch.rand(x.shape[0], n, device=x.device) >= self.vs_dropout).to(x.dtype)
            keep = drop if keep is None else keep * drop
        if self.use_norm:
            mu = x.mean(1, keepdim=True).detach()
            x = x - mu
            sd = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / sd
        if self.cycle is not None:
            if phase is None:
                raise ValueError("rcf_cycle needs the window phase")
            c = self.cycle(phase, 0, self.seq_len, var_idx)
            x = (x - (c if keep is None else c * keep[:, None])
                 - self.cycle.shared_part(phase, 0, self.seq_len))
        h = self.embedding(x, x_mark)
        if self.id_loc == "input":
            h = self.identity[0](h, var_idx, keep)
        for i, layer in enumerate(self.layers):
            if self.id_loc == "every":
                h = self.identity[i](h, var_idx, keep)
            h = layer(h, var_idx, keep)
        out = self.projector(apply_norm(self.norm, h, var_idx, keep)).permute(0, 2, 1)[:, :, :n]
        if self.cycle is not None:
            c = self.cycle(phase, self.seq_len, self.pred_len, var_idx)
            out = (out + (c if keep is None else c * keep[:, None])
                   + self.cycle.shared_part(phase, self.seq_len, self.pred_len))
        if self.use_norm:
            out = out * sd + mu
        return out

    def set_eval_seed(self, seed):
        """Evaluate S-Mamba in one random scan order drawn with this seed (None = original)."""
        for layer in self.layers:
            if isinstance(layer, MambaLayer):
                layer.eval_seed = seed

    def variable_specific_parameters(self):
        """Zero-initialised parameters besides the identity tables: channel-norm offsets and
        cycles (including a shared cycle); they get their own learning-rate multiplier."""
        return [p for name, p in self.named_parameters()
                if name.endswith(("d_weight", "d_bias")) or name.startswith("cycle.")]

    def identity_weight(self):
        """The input-level identity table (N, D), or None."""
        return self.identity[0].weight()

    def set_eval_orders(self, k):
        """Number of random scan orders averaged at evaluation (S-Mamba; 1 = original order)."""
        for layer in self.layers:
            if isinstance(layer, MambaLayer):
                layer.eval_orders = k

    def mamba_backend(self):
        """'cuda' (mamba_ssm kernel), 'torch' (MambaTorch) or None (no Mamba layers)."""
        for layer in self.layers:
            if isinstance(layer, MambaLayer):
                return "torch" if isinstance(layer.fwd, MambaTorch) else "cuda"
        return None
