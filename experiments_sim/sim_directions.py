"""Sandbox simulations for the feasibility derivations in papers/theory_directions.md
(newpaper repository). Synthetic multivariate series with known structure, linear / small
neural forecasters, CPU only.

python experiments_sim/sim_directions.py [A|A2|B|D|E|all] [--seeds 3]

Data: N variables, hourly, daily cycle P = 24. Each variable is
    x_it = mu_i + s_i * (shared profile + tau * own profile + AR(1) residual [+ loading * factor])
Drift: a fraction rho of the variables changes at a random time (in the test period, or in
validation + test): a new own profile ("profile", hard to see in window statistics) or a new
profile plus a level / scale shift ("level", visible in window statistics).
Forecast: look-back L = 48, horizon H = 24, instance normalisation per window, MSE in
training-standardised units.
"""
import argparse
import json
import math
import sys
import numpy as np

P, L, H = 24, 48, 24
PHASE = [24]          # phase cycle of the calendar features / local corrections (168 with weekly)


# ----------------------------------------------------------------------------- data
def make_data(rng, N=60, days=200, tau=1.0, rho=0.0, drift="level", timing="test", factor=0.0,
              weekly=0.0):
    """weekly > 0 adds a variable-specific weekly profile (period 168 h > look-back 48 h, so it
    cannot be read off the window: an identity gap in the sense of theory Proposition 1)."""
    T = days * P
    hours = np.arange(T) % P
    how = np.arange(T) % 168
    grid = np.arange(P)

    def own_profile(n=P):
        g = np.arange(n)
        out = np.zeros(n)
        for k in (1, 2, 3):
            a, b = rng.normal(size=2) / math.sqrt(3)
            out += a * np.sin(2 * np.pi * k * g / n) + b * np.cos(2 * np.pi * k * g / n)
        return out

    shared = np.sin(2 * np.pi * grid / P) + 0.5 * np.sin(4 * np.pi * grid / P)
    n_tr, n_va = int(0.6 * T), int(0.2 * T)
    r = np.zeros((T, N))
    eps = rng.normal(scale=0.4, size=(T, N))
    for t in range(1, T):
        r[t] = 0.7 * r[t - 1] + eps[t]
    f = np.zeros(T)
    if factor:
        e_f = rng.normal(scale=0.3, size=T)
        for t in range(1, T):
            f[t] = 0.95 * f[t - 1] + e_f[t]
    lam = rng.uniform(0.5, 1.5, size=N) * factor
    mu, sg = rng.normal(scale=2.0, size=N), np.exp(rng.normal(scale=0.5, size=N))
    prof = np.stack([shared + tau * own_profile() for _ in range(N)], 1)        # (P, N)
    week = np.stack([weekly * own_profile(168) for _ in range(N)], 1)          # (168, N)
    x = mu + sg * (prof[hours] + week[how] + r + lam * f[:, None])
    tc = np.full(N, T)                                                         # change times
    n_drift = int(round(rho * N))
    first = n_tr + n_va if timing == "test" else n_tr
    for i in rng.choice(N, n_drift, replace=False):
        tc[i] = rng.integers(first, T - 5 * P)
        p2 = shared + tau * own_profile()
        m2, s2 = mu[i], sg[i]
        if drift == "level":
            m2 = mu[i] + 3 * sg[i] * rng.choice([-1, 1])
            s2 = sg[i] * rng.choice([0.5, 2.0])
        seg = slice(tc[i], T)
        w2 = weekly * own_profile(168)
        x[seg, i] = m2 + s2 * (p2[hours[seg]] + w2[how[seg]] + r[seg, i])      # loading -> 0
    m, s = x[:n_tr].mean(0), x[:n_tr].std(0)
    return (x - m) / s, tc, n_tr, n_va


def windows(x, lo, hi):
    """Windows whose targets lie in [lo, hi): X (S, N, L), Y (S, N, H), phase (S,), end (S,)."""
    starts = np.arange(max(lo - L, 0), hi - L - H + 1)
    idx = starts[:, None] + np.arange(L + H)
    w = x[idx]                                                                 # (S, L+H, N)
    return (w[:, :L].transpose(0, 2, 1), w[:, L:].transpose(0, 2, 1), (starts + L) % PHASE[0],
            starts + L)


def split_all(x, n_tr, n_va):
    T = len(x)
    return {"train": windows(x, 0, n_tr), "val": windows(x, n_tr, n_tr + n_va),
            "test": windows(x, n_tr + n_va, T)}


def inorm(X, Y):
    m, s = X.mean(-1, keepdims=True), X.std(-1, keepdims=True) + 1e-5
    return (X - m) / s, (Y - m) / s, m, s


def window_stats(X):
    return np.stack([X.mean(-1), np.log(X.std(-1) + 1e-5)], -1)               # (S, N, 2)


def ridge(A, B, lam=1e-1):
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ B)


def onehot(ph):
    return np.eye(PHASE[0])[ph]


# ----------------------------------------------------------------------------- forecasters
def fit_global(tr, cross=False):
    Xn, Yn, _, _ = inorm(tr[0], tr[1])
    S, N, _ = Xn.shape
    feats = [Xn, np.repeat(onehot(tr[2])[:, None], N, 1)]
    if cross:
        feats.append(np.repeat(Xn.mean(1, keepdims=True), N, 1))
    A = np.concatenate(feats, -1).reshape(S * N, -1)
    return ridge(A, Yn.reshape(S * N, H))


def predict_global(W, sp, cross=False, weights=None):
    Xn, _, m, s = inorm(sp[0], sp[1])
    S, N, _ = Xn.shape
    feats = [Xn, np.repeat(onehot(sp[2])[:, None], N, 1)]
    if cross:
        if weights is None:
            c = Xn.mean(1, keepdims=True)
        else:                                       # reliability-weighted cross-section
            w = weights / weights.sum(1, keepdims=True)
            c = (w[..., None] * Xn).sum(1, keepdims=True)
        feats.append(np.repeat(c, N, 1))
    Gn = (np.concatenate(feats, -1).reshape(S * N, -1) @ W).reshape(S, N, H)
    return Gn, m, s


def fit_local_bias(tr, W, cross=False, shrink=5.0):
    """Per-variable, per-phase correction of the global forecast (identity x phase term)."""
    Gn, _, _ = predict_global(W, tr, cross)
    _, Yn, _, _ = inorm(tr[0], tr[1])
    N = Yn.shape[1]
    b = np.zeros((N, PHASE[0], H))
    for ph in range(PHASE[0]):
        sel = tr[2] == ph
        b[:, ph] = (Yn[sel] - Gn[sel]).sum(0) / (sel.sum() + shrink)
    return b


def train_stats(tr):
    st = window_stats(tr[0])                                                   # (S, N, 2)
    mu = st.mean(0)
    d = st - mu
    cov = np.einsum("sni,snj->nij", d, d) / (len(st) - 1) + 1e-4 * np.eye(2)
    return mu, np.linalg.inv(cov)


def mahalanobis(X, mu, icov):
    d = window_stats(X) - mu
    return np.einsum("sni,nij,snj->sn", d, icov, d)


CHI2 = {p: -2 * math.log(1 - p) for p in (0.9, 0.95, 0.99, 0.995, 0.999)}


def mse(a, b, mask=None):
    e = ((a - b) ** 2).mean(-1)                                                 # (S, N)
    return float(e.mean()) if mask is None else (float(e[mask].mean()) if mask.any() else float("nan"))


def auc(score, label):
    if label.all() or not label.any():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = label.sum()
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(score) - n1)))


# ----------------------------------------------------------------------------- A: combination
def sim_A(seed, tau, rho, drift, timing, return_parts=False, weekly=0.0):
    rng = np.random.default_rng(seed)
    PHASE[0] = 168 if weekly else 24
    x, tc, n_tr, n_va = make_data(rng, tau=tau, rho=rho, drift=drift, timing=timing, weekly=weekly)
    sp = split_all(x, n_tr, n_va)
    W = fit_global(sp["train"])
    b = fit_local_bias(sp["train"], W)
    mu_s, icov = train_stats(sp["train"])
    out = {}
    pred = {}
    for name in ("val", "test"):
        X, Y, ph, end = sp[name]
        Gn, m, s = predict_global(W, sp[name])
        G = Gn * s + m
        Lp = (Gn + b[np.arange(b.shape[0])[None, :], ph[:, None]]) * s + m
        pred[name] = (G, Lp, Y, mahalanobis(X, mu_s, icov), end)
    G, Lp, Y, _, _ = pred["val"]
    e, d = Y - G, Lp - G
    w_bg = np.clip((e * d).sum((0, 2)) / ((d * d).sum((0, 2)) + 1e-12), 0, 1)  # (N,)

    def combine(split, w):
        G, Lp, _, _, _ = pred[split]
        return G + w[..., None] * (Lp - G)

    def post_w(split, c):
        D2 = pred[split][3]
        q = np.ones_like(D2) if c is None else 1 / (1 + np.exp(-np.clip((c - D2) / 2, -50, 50)))
        return q * w_bg[None]

    # val-selected and fixed (false-alarm 0.1 %) thresholds
    cands = [None] + list(CHI2.values())
    val_mse = [mse(combine("val", post_w("val", c)), pred["val"][2]) for c in cands]
    c_val = cands[int(np.argmin(val_mse))]
    # hard global gate (|mean| + |log std|), threshold = quantile of training scores, val-selected
    st_tr = window_stats(sp["train"][0])
    score_tr = np.abs(st_tr[..., 0]) + np.abs(st_tr[..., 1])
    hard_cands = [None] + [np.quantile(score_tr, q) for q in (0.9, 0.95, 0.98, 0.99, 0.995, 0.999)]

    def hard_w(split, t):
        st = window_stats(sp[split][0])
        sc = np.abs(st[..., 0]) + np.abs(st[..., 1])
        return np.ones_like(sc) if t is None else (sc <= t).astype(float)
    hv = [mse(combine("val", hard_w("val", t)), pred["val"][2]) for t in hard_cands]
    t_val = hard_cands[int(np.argmin(hv))]

    G, Lp, Y, D2, end = pred["test"]
    post = end[:, None] > tc[None, :]                                         # input after change
    stable_var = np.repeat((tc >= len(x))[None], len(end), 0)
    methods = {
        "global": G, "local": Lp,
        "static_bg": combine("test", np.repeat(w_bg[None], len(end), 0)),
        "posterior_val": combine("test", post_w("test", c_val)),
        "posterior_fixed": combine("test", post_w("test", CHI2[0.999])),
        "hard_val": combine("test", hard_w("test", t_val)),
        "oracle_regime": combine("test", w_bg[None] * (~post)),
    }
    for k, v in methods.items():
        out[k] = {"all": mse(v, Y), "drifted": mse(v, Y, post), "stable_vars": mse(v, Y, stable_var)}
    out["auc_D2"] = auc(D2[stable_var | post], post[stable_var | post])
    out["c_val"] = None if c_val is None else round(c_val, 2)
    out["mean_w_bg"] = float(w_bg.mean())
    # theorem A1 check: predicted gain of the oracle-regime weight over the static weight
    e, d = Y - G, Lp - G
    ed, dd = (e * d).sum(-1), (d * d).sum(-1)
    grp = post.astype(int)
    pred_gain = 0.0
    for g in (0, 1):
        sel = grp == g
        if sel.any():
            pred_gain += (ed[sel].sum() ** 2 / dd[sel].sum()) / ed.size
    pred_gain -= (ed.sum() ** 2 / dd.sum()) / ed.size
    out["theorem_gain_regime_vs_static_opt"] = float(pred_gain / H)
    if return_parts:
        return out, pred, w_bg, c_val, sp, tc, x
    return out


# ----------------------------------------------------------------------------- A2: messages
def sim_A2(seed, rho):
    rng = np.random.default_rng(seed)
    x, tc, n_tr, n_va = make_data(rng, tau=1.0, rho=rho, drift="level", timing="test", factor=1.0)
    sp = split_all(x, n_tr, n_va)
    W_ci, W_cd = fit_global(sp["train"]), fit_global(sp["train"], cross=True)
    mu_s, icov = train_stats(sp["train"])
    X, Y, _, end = sp["test"]
    q = 1 / (1 + np.exp(-(CHI2[0.999] - mahalanobis(X, mu_s, icov)) / 2))
    post = end[:, None] > tc[None, :]
    stable = np.repeat((tc >= len(x))[None], len(end), 0)
    res, preds = {}, {}
    for name, (Wm, cross, wts) in {"channel_independent": (W_ci, False, None),
                                    "cross_uniform": (W_cd, True, None),
                                    "cross_reliability": (W_cd, True, q),
                                    "cross_oracle": (W_cd, True, (~post).astype(float) + 1e-9)}.items():
        Gn, m, s = predict_global(Wm, sp["test"], cross, wts)
        res[name] = {"stable_vars": mse(Gn * s + m, Y, stable), "all": mse(Gn * s + m, Y)}
        preds[name] = Gn * s + m
    # both gates: reliability-weighted messages for the others, output fallback for the drifting
    # variable itself (q * cross_reliability + (1 - q) * channel_independent)
    both = q[..., None] * preds["cross_reliability"] + (1 - q[..., None]) * preds["channel_independent"]
    res["cross_reliability+self_gate"] = {"stable_vars": mse(both, Y, stable), "all": mse(both, Y)}
    return res


# ----------------------------------------------------------------------------- B: conformal
def sim_B(seed, timing, alpha=0.1, weekly=0.0):
    out, pred, w_bg, c_val, sp, tc, x = sim_A(seed, 1.0, 0.2, "level", timing, return_parts=True,
                                              weekly=weekly)

    def forecast(split):
        G, Lp, Y, D2, end = pred[split]
        q = 1 / (1 + np.exp(-(CHI2[0.999] - D2) / 2))
        return G + (q * w_bg[None])[..., None] * (Lp - G), Y, D2, end

    Fv, Yv, Dv, _ = forecast("val")
    Ft, Yt, Dt, end = forecast("test")
    rv, rt = np.abs(Yv - Fv), np.abs(Yt - Ft)                                 # (S, N, H)
    post = end[:, None] > tc[None, :]
    stable = np.repeat((tc >= len(x))[None], len(end), 0)
    level = math.ceil((1 - alpha) * (rv.shape[0] * rv.shape[1] + 1)) / (rv.shape[0] * rv.shape[1])
    qm = np.quantile(rv.reshape(-1, H), min(level, 1.0), axis=0)               # marginal, per step
    qv = np.quantile(rv, 1 - alpha, axis=0)                                    # per variable
    c0 = CHI2[0.99]
    gv, gt = Dv > c0, Dt > c0
    q_hi = np.quantile(rv[gv], 1 - alpha, axis=0) if gv.sum() >= 50 else qm
    q_lo = np.quantile(rv[~gv], 1 - alpha, axis=0)
    widths = {"marginal": np.broadcast_to(qm, rt.shape), "per_variable": np.broadcast_to(qv, rt.shape),
              "mondrian_D2": np.where(gt[..., None], q_hi, q_lo)}
    res = {"n_cal_high_D2": int(gv.sum())}
    for k, wdt in widths.items():
        cov = rt <= wdt
        res[k] = {"coverage_all": float(cov.mean()), "coverage_drifted": float(cov[post].mean()),
                  "coverage_stable_vars": float(cov[stable].mean()), "width": float(wdt.mean())}
    return res


# ----------------------------------------------------------------------------- D / E: neural
def neural_data(seed, heavy=False):
    rng = np.random.default_rng(seed)
    x, tc, n_tr, n_va = make_data(rng, N=40, days=160, tau=1.0, rho=0.0)
    if heavy:                                   # one variable with large noise dominates the loss
        noise = np.random.default_rng(seed + 99).normal(scale=4.0, size=len(x))
        x[:, 0] = x[:, 0] + noise
    sp = split_all(x, n_tr, n_va)

    def tens(s):
        Xn, Yn, m, sd = inorm(s[0], s[1])
        S, N, _ = Xn.shape
        feats = np.concatenate([Xn, np.repeat(onehot(s[2])[:, None], N, 1),
                                np.repeat(np.eye(N)[None], S, 0)], -1)          # + identity one-hot
        return feats.reshape(S * N, -1), Yn.reshape(S * N, H), m.reshape(S * N, 1), sd.reshape(S * N, 1), \
            np.repeat(np.arange(N)[None], S, 0).reshape(-1), s[1].reshape(S * N, H)
    return {k: tens(v) for k, v in sp.items()}


def sim_D(seed, K=5, epochs=6):
    import torch
    torch.set_num_threads(4)
    data = neural_data(seed)
    Xtr, Ytr = (torch.tensor(a, dtype=torch.float32) for a in data["train"][:2])
    Xte, _, mte, ste, _, Yraw = data["test"]
    Xte = torch.tensor(Xte, dtype=torch.float32)

    def net():
        return torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 256), torch.nn.GELU(),
                                   torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, H))

    def train(model, ep, data_seed, ema=None, swa=None):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(data_seed)
        for e in range(ep):
            for idx in torch.randperm(len(Xtr), generator=g).split(512):
                loss = ((model(Xtr[idx]) - Ytr[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if ema is not None:
                    with torch.no_grad():
                        for pe, p in zip(ema.parameters(), model.parameters()):
                            pe.mul_(0.99).add_(p, alpha=0.01)
            if swa is not None and e >= ep // 2:
                swa.append([p.detach().clone() for p in model.parameters()])
        return model

    def pred(model):
        with torch.no_grad():
            return model(Xte).numpy() * ste + mte

    def avg_into(model, plist):
        with torch.no_grad():
            for i, p in enumerate(model.parameters()):
                p.copy_(torch.stack([pl[i] for pl in plist]).mean(0))
        return model

    err = lambda pr: float(((pr - Yraw) ** 2).mean())
    singles, preds, params, ema_err, swa_err = [], [], [], [], []
    for k in range(K):
        torch.manual_seed(1000 * seed + k)
        m = net()
        torch.manual_seed(1000 * seed + k)
        ema = net()
        ema.load_state_dict(m.state_dict())
        swa = []
        train(m, epochs, 1000 * seed + k, ema, swa)
        preds.append(pred(m))
        singles.append(err(preds[-1]))
        params.append([p.detach().clone() for p in m.parameters()])
        ema_err.append(err(pred(ema)))
        swa_err.append(err(pred(avg_into(net(), swa))))
    P_ = np.stack(preds)
    v_seed = float(P_.var(0).mean())
    res = {"single": float(np.mean(singles)), "output_ensemble": err(P_.mean(0)),
           "weight_avg_from_scratch": err(pred(avg_into(net(), params))),
           "ema": float(np.mean(ema_err)), "swa_one_run": float(np.mean(swa_err)),
           "seed_variance": v_seed, "predicted_ensemble_gain": v_seed * (1 - 1 / K)}
    # soup from a shared starting point: train 3 epochs, fork K copies with different data orders
    torch.manual_seed(1000 * seed)
    base = train(net(), epochs // 2, 1000 * seed)
    forks, fp = [], []
    for k in range(K):
        mk = net()
        mk.load_state_dict(base.state_dict())
        train(mk, epochs - epochs // 2, 77 + k)
        forks.append(pred(mk))
        fp.append([p.detach().clone() for p in mk.parameters()])
    res["shared_start_single"] = float(np.mean([err(f) for f in forks]))
    res["shared_start_output_ensemble"] = err(np.mean(forks, 0))
    res["shared_start_soup"] = err(pred(avg_into(net(), fp)))
    res["actual_ensemble_gain"] = res["single"] - res["output_ensemble"]
    return res


def sim_E(seed, epochs=6):
    import torch
    torch.set_num_threads(4)
    data = neural_data(seed, heavy=True)
    Xtr, Ytr = (torch.tensor(a, dtype=torch.float32) for a in data["train"][:2])
    vtr = torch.tensor(data["train"][4])
    Xte, _, mte, ste, vte, Yraw = data["test"]
    Xte = torch.tensor(Xte[:, :L + P], dtype=torch.float32)                    # shared model: no identity
    Xtr = Xtr[:, :L + P]
    N = int(vtr.max()) + 1
    res = {}
    for obj in ("erm", "normalised", "group_dro"):
        torch.manual_seed(seed)
        m = torch.nn.Sequential(torch.nn.Linear(L + P, 256), torch.nn.GELU(), torch.nn.Linear(256, H))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(seed)
        q = torch.ones(N) / N
        scale = torch.ones(N)
        for e in range(epochs):
            for idx in torch.randperm(len(Xtr), generator=g).split(512):
                le = ((m(Xtr[idx]) - Ytr[idx]) ** 2).mean(1)
                v = vtr[idx]
                per = torch.zeros(N).index_add_(0, v, le) / torch.bincount(v, minlength=N).clamp(min=1)
                if obj == "erm":
                    loss = le.mean()
                elif obj == "normalised":
                    loss = (le / scale[v]).mean()
                else:
                    with torch.no_grad():
                        q = q * torch.exp(0.01 * per)
                        q = q / q.sum()
                    loss = (q * per).sum() * N / N
                opt.zero_grad()
                loss.backward()
                opt.step()
            if obj == "normalised":                  # per-variable error scale from the last epoch
                with torch.no_grad():
                    le = ((m(Xtr) - Ytr) ** 2).mean(1)
                    scale = (torch.zeros(N).index_add_(0, vtr, le) / torch.bincount(vtr, minlength=N)).clamp(min=1e-3)
        with torch.no_grad():
            pr = m(Xte).numpy() * ste + mte
        e_var = np.array([((pr[vte == i] - Yraw[vte == i]) ** 2).mean() for i in range(N)])
        res[obj] = {"avg_mse": float(e_var.mean()), "avg_mse_without_heavy": float(e_var[1:].mean()),
                    "median_var_mse": float(np.median(e_var))}
    return res


# ----------------------------------------------------------------------------- driver
def mean_dict(ds):
    if isinstance(ds[0], dict):
        return {k: mean_dict([d[k] for d in ds]) for k in ds[0]}
    vals = [d for d in ds if d is not None and not (isinstance(d, float) and math.isnan(d))]
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = range(a.seeds)
    results = {}
    if a.which in ("A", "all"):
        grid = [(tau, rho, drift, timing) for tau in (0.0, 1.0) for rho in (0.0, 0.1, 0.3)
                for drift in ("level", "profile") for timing in ("test", "valtest")
                if not (rho == 0.0 and (drift, timing) != ("level", "test"))]
        results["A"] = {f"tau={t} rho={r} drift={d} timing={tm}":
                        mean_dict([sim_A(s, t, r, d, tm) for s in seeds]) for t, r, d, tm in grid}
        results["A_weekly"] = {f"weekly={wk} rho={r} timing={tm}":
                               mean_dict([sim_A(s, 1.0, r, "level", tm, weekly=wk) for s in seeds])
                               for wk in (0.0, 1.0, 2.0) for r in (0.0, 0.1, 0.3)
                               for tm in (("test",) if r == 0 else ("test", "valtest"))}
    if a.which in ("A2", "all"):
        results["A2"] = {f"rho={r}": mean_dict([sim_A2(s, r) for s in seeds]) for r in (0.0, 0.1, 0.3)}
    if a.which in ("B", "all"):
        results["B"] = {f"weekly={wk} timing={tm}": mean_dict([sim_B(s, tm, weekly=wk) for s in seeds])
                        for wk in (0.0, 1.0) for tm in ("valtest", "test")}
    if a.which in ("D", "all"):
        results["D"] = mean_dict([sim_D(s) for s in seeds])
    if a.which in ("E", "all"):
        results["E"] = mean_dict([sim_E(s) for s in seeds])
    text = json.dumps(results, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    sys.exit(main())
