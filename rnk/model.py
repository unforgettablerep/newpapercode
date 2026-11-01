"""Reliability-gated node knowledge (RNK) on a global forecaster
(newpaper papers/model_direction.md, section 2).

All quantities are in units of the sample's load scale sigma (data.py). For a sample of node i:

  knowledge   k = g_d(desc) + g_p(profile) + delta_i                              (NKE)
              delta_i is a small-initialised identity that only training nodes have
  gating      (KRG) q_id in [0, 1] per sample is the posterior that the node is still in the
              state its persistence-dependent knowledge was estimated in: it scales the identity
              (q_id delta_i) and mixes the R-day profile with the short profile,
              q_id prof + (1 - q_id) prof_s. The descriptors are documented and stay. For
              ablation, q in [0, 1] shrinks the whole knowledge: k_eff = q k + (1 - q) k_null
  covariates  u_t = sum_r (beta0 + W k_eff)_r s_r(cov_t, cal_t)                  (KCD)
                    + sum_k capw_k(t) a_k(cov_t, cal_t)
              s_r: shared basis functions; a_k: response per unit of capacity of technology k,
              a_k = scale_k * softplus(net_k) ("mono": one sign for all covariate values, e.g. PV
              output can only lower the load) or scale_k * net_k ("free", e.g. batteries);
              beta and scale_k start at zero, so u starts at zero
  forecast    the backbone sees the look-back with u removed (and RevIN), forecasts the base,
              and u is added back over the target hours; the knowledge enters the backbone by
              FiLM on its first hidden layer, zero-initialised

With every part switched off the model is the plain global backbone, and at initialisation the
full model computes exactly what the plain backbone computes (tests/test_rnk.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGNS = {"mono": F.softplus, "free": lambda z: z}


class MLPBackbone(nn.Module):
    """Global MLP over the normalised look-back, past/future covariates and calendar."""

    def __init__(self, L, H, C, Cc, hidden=512, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(L + L * C + L * Cc + H * C + H * Cc, hidden)
        self.hid = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, H)
        self.drop = nn.Dropout(dropout)
        self.hidden = hidden

    def forward(self, xn, covp, calp, covf, calf, film=None):
        z = torch.cat([xn, covp.flatten(1), calp.flatten(1), covf.flatten(1), calf.flatten(1)], 1)
        h = F.gelu(self.inp(z))
        if film is not None:
            g, b = film
            h = h * (1 + g) + b
        return self.out(self.drop(F.gelu(self.hid(h))))


class KnowledgeEncoder(nn.Module):
    def __init__(self, n_nodes, desc_dim, prof_dim, d_k=32, hidden=64, use_desc=True,
                 use_prof=True, use_id=True):
        super().__init__()
        mlp = lambda d: nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d_k))
        self.desc = mlp(desc_dim) if use_desc else None
        self.prof = mlp(prof_dim) if use_prof else None
        # small random start: the identity enters through a zero-initialised FiLM, and a zero
        # identity times a zero FiLM weight would get no gradient at all
        self.ident = nn.Parameter(0.02 * torch.randn(n_nodes, d_k)) if use_id else None
        self.null = nn.Parameter(torch.zeros(d_k))
        self.d_k = d_k

    def forward(self, node, desc, prof, id_known=None, prof_s=None, q_prof=None):
        """node: (B,) index; id_known: (B,) 1 where the node has a trained identity; q_prof: (B,)
        weight of the R-day profile against the short profile prof_s."""
        k = torch.zeros(len(node), self.d_k, device=node.device)
        if self.desc is not None:
            k = k + self.desc(desc)
        if self.prof is not None:
            if q_prof is not None and prof_s is not None:
                prof = q_prof[:, None] * prof + (1 - q_prof[:, None]) * prof_s
            k = k + self.prof(prof)
        if self.ident is not None:
            idv = self.ident[node.clamp(0, self.ident.shape[0] - 1)]
            if id_known is not None:
                idv = idv * id_known[:, None]
            k = k + idv
        return k


class CovariateResponse(nn.Module):
    """u over the look-back and target hours (B, L + H)."""

    def __init__(self, f_dim, d_k, n_basis=8, n_cap=0, cap_signs=(), hidden=32, use_basis=True,
                 knowledge=True):
        super().__init__()
        self.basis = (nn.Sequential(nn.Linear(f_dim, hidden), nn.GELU(), nn.Linear(hidden, n_basis))
                      if use_basis else None)
        self.beta0 = nn.Parameter(torch.zeros(n_basis)) if use_basis else None
        self.beta = nn.Linear(d_k, n_basis, bias=False) if (use_basis and knowledge) else None
        if self.beta is not None:
            nn.init.zeros_(self.beta.weight)
        self.cap = nn.ModuleList(nn.Sequential(nn.Linear(f_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
                                 for _ in range(n_cap))
        self.cap_scale = nn.Parameter(torch.zeros(n_cap))
        self.signs = list(cap_signs)

    def forward(self, feats, k_eff, capw):
        """feats: (B, L + H, f_dim); k_eff: (B, d_k) or None; capw: (B, L + H, K)."""
        u = torch.zeros(feats.shape[:2], device=feats.device)
        if self.basis is not None:
            beta = self.beta0.expand(len(feats), -1)
            if self.beta is not None and k_eff is not None:
                beta = beta + self.beta(k_eff)
            u = u + (self.basis(feats) * beta[:, None]).sum(-1)
        for j, net in enumerate(self.cap):
            a = SIGNS[self.signs[j]](net(feats).squeeze(-1)) * self.cap_scale[j]
            u = u + capw[..., j] * a
        return u


class RNKForecaster(nn.Module):
    def __init__(self, L, H, C, Cc, K, n_nodes, desc_dim, prof_dim=168, hidden=512, d_k=32,
                 use_id=False, use_desc=False, use_prof=False, use_basis=False, use_cap=False,
                 cap_signs=None, n_basis=8, dropout=0.1, revin=True):
        super().__init__()
        self.L, self.H, self.revin = L, H, revin
        self.backbone = MLPBackbone(L, H, C, Cc, hidden, dropout)      # created first: same RNG
        self.uses_knowledge = use_id or use_desc or use_prof
        self.know = self.film = None
        if self.uses_knowledge:
            self.know = KnowledgeEncoder(n_nodes, desc_dim, prof_dim, d_k, use_desc=use_desc,
                                         use_prof=use_prof, use_id=use_id)
            self.film = nn.Linear(d_k, 2 * hidden)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.resp = None
        if use_basis or use_cap:
            signs = list(cap_signs or ["free"] * K) if use_cap else []
            self.resp = CovariateResponse(C + Cc, d_k, n_basis, len(signs), signs,
                                          use_basis=use_basis, knowledge=self.uses_knowledge)

    def knowledge(self, b, q_id=None):
        if not self.uses_knowledge:
            return None
        idk = b.get("id_known")
        if q_id is not None:
            idk = q_id if idk is None else idk * q_id
        return self.know(b["node"], b["desc"], b["prof"], idk, b.get("prof_s"), q_id)

    def gate(self, k, q):
        if k is None or q is None:
            return k
        return q[:, None] * k + (1 - q[:, None]) * self.know.null

    def decompose(self, b, k_eff):
        if self.resp is None:
            return torch.zeros(len(b["x"]), self.L + self.H, device=b["x"].device)
        feats = torch.cat([torch.cat([b["covp"], b["covf"]], 1), torch.cat([b["calp"], b["calf"]], 1)], 2)
        return self.resp(feats, k_eff, b["capw"])

    def forward(self, b, q=None, q_id=None, return_parts=False):
        k_eff = self.gate(self.knowledge(b, q_id), q)
        u = self.decompose(b, k_eff)
        xt = b["x"] - u[:, : self.L]
        if self.revin:
            m = xt.mean(1, keepdim=True)
            s = torch.sqrt(xt.var(1, keepdim=True, unbiased=False) + 1e-5)
        else:
            m, s = torch.zeros_like(xt[:, :1]), torch.ones_like(xt[:, :1])
        film = self.film(k_eff).chunk(2, -1) if k_eff is not None else None
        base = self.backbone((xt - m) / s, b["covp"], b["calp"], b["covf"], b["calf"], film)
        yhat = base * s + m + u[:, self.L:]
        if return_parts:
            return yhat, xt, u
        return yhat
