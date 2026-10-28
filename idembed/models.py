"""iTransformer and S-Mamba (variate-token forecasters) with variable identity embeddings.

Both follow the official code bases: series-wise instance normalisation, one token per
variable (plus one per calendar feature) from Linear(seq_len -> d_model), e_layers encoder
layers, a final LayerNorm and Linear(d_model -> pred_len). The only change is the optional
identity embedding added to the variable tokens, either once after the embedding ('input')
or before every encoder layer ('every').
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .identity import VariableIdentity
from .mamba import build_mamba


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

    def __init__(self, d_model, n_heads, d_ff, dropout, activation):
        super().__init__()
        self.attn = FullAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        return self.norm2(x + self.ffn(x))


class MambaLayer(nn.Module):
    """S-Mamba EncoderLayer: a forward and a reversed Mamba over the token sequence, summed;
    x = LN(x + Mamba(x) + flip(Mamba_r(flip(x)))); x = LN(x + FFN(x))."""

    def __init__(self, d_model, d_state, d_ff, dropout, activation, mamba_impl="auto"):
        super().__init__()
        self.fwd = build_mamba(d_model, d_state, d_conv=2, expand=1, impl=mamba_impl)
        self.bwd = build_mamba(d_model, d_state, d_conv=2, expand=1, impl=mamba_impl)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.fwd(x) + self.bwd(x.flip(1)).flip(1))
        return self.norm2(x + self.ffn(x))


class VariateForecaster(nn.Module):
    def __init__(self, backbone, n_vars, seq_len, pred_len, d_model=512, d_ff=512, e_layers=2,
                 n_heads=8, d_state=16, dropout=0.1, activation="gelu", use_norm=True,
                 identity="none", id_rank=8, id_loc="input", id_dropout=0.0, mamba_impl="auto"):
        super().__init__()
        self.use_norm, self.pred_len, self.id_loc = use_norm, pred_len, id_loc
        self.embedding = DataEmbeddingInverted(seq_len, d_model, dropout)
        if backbone == "itransformer":
            make = lambda: TransformerLayer(d_model, n_heads, d_ff, dropout, activation)
        elif backbone == "smamba":
            make = lambda: MambaLayer(d_model, d_state, d_ff, dropout, activation, mamba_impl)
        else:
            raise ValueError(backbone)
        self.layers = nn.ModuleList(make() for _ in range(e_layers))
        self.norm = nn.LayerNorm(d_model)
        self.projector = nn.Linear(d_model, pred_len)
        n_id = 1 if id_loc == "input" else e_layers
        if id_loc not in ("input", "every"):
            raise ValueError(id_loc)
        # created last so that the other layers get exactly the original initialisation
        self.identity = nn.ModuleList(VariableIdentity(n_vars, d_model, identity, id_rank, id_dropout)
                                      for _ in range(n_id))

    def forward(self, x, x_mark=None, var_idx=None):
        """x: (B, L, n) with the variables listed in var_idx (default: all, in order)."""
        n = x.shape[-1]
        if self.use_norm:
            mu = x.mean(1, keepdim=True).detach()
            x = x - mu
            sd = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / sd
        h = self.embedding(x, x_mark)
        if self.id_loc == "input":
            h = self.identity[0](h, var_idx)
        for i, layer in enumerate(self.layers):
            if self.id_loc == "every":
                h = self.identity[i](h, var_idx)
            h = layer(h)
        out = self.projector(self.norm(h)).permute(0, 2, 1)[:, :, :n]
        if self.use_norm:
            out = out * sd + mu
        return out

    def identity_weight(self):
        """The input-level identity table (N, D), or None."""
        return self.identity[0].weight()
