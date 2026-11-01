"""Training and evaluation of RNKForecaster on the sample arrays of data.py.

Training: Adam, MSE on the target in units of sigma (missing targets masked), learning rate
halved every `decay_every` epochs; the weights are averaged with an exponential moving average
(decay `ema` per step) and the averaged model is validated after every epoch (knowledge
trusted); early stopping with patience `patience` after at least `min_epochs` epochs; the best
averaged weights are kept. Persistence dropout `kd_id`:
each training sample gets q_id = 0 with this probability (no identity, short profile; see
model.py), which trains the fallback the gate uses (theory.md section 4). Knowledge dropout
`kd` (ablation): the whole knowledge is replaced by the null knowledge (q = 0).

Evaluation modes (one trained model; the gate acts through q_id unless noted):
  trust     q_id = 1: identity and R-day profile
  noid      q_id = 0: the fallback (no identity, short profile; descriptors and KCD stay)
  null      no knowledge at all (q = 0)
  gate      label-free: q_id = sigmoid((c - D2) / 2), c = chi-square(4) quantile at 0.999
  gate_val  the same with c chosen on the validation split among {none, 0.9 .. 0.999}
  gate_err  delayed labels (actuals arrive daily): forecast = f0 + w (f1 - f0), f1 = trust,
            f0 = noid, w = clip(A / B, 0, 1) with exponentially weighted (half-life 7 days)
            A = sum e d, B = sum d^2 over the days before the forecast day, e = y - f0,
            d = f1 - f0; this is the optimal weight of theorem A1 estimated from recent days,
            started on the validation days
  gate_comb f0 + w q (f1 - f0): the delayed-error weight times the label-free posterior
  oracle    synthetic data only: q_id = 0 after a node's undocumented change, else 1
"""
import copy
import math
import time

import numpy as np
import torch

from .gate import NodeReference, chi2_threshold, posterior_q, window_stats

KEYS = ("x", "yf", "ymask", "covp", "covf", "calp", "calf", "hodp", "prof", "prof_s", "desc", "capw",
        "node", "id_known", "day", "xprof_l", "xprof_s")
GATE_PS = (0.9, 0.95, 0.99, 0.995, 0.999)


def to_torch(S, device="cpu"):
    out = {}
    for k in KEYS:
        if k not in S:
            continue
        v = S[k]
        dt = torch.long if k in ("node", "hodp", "day") else torch.float32
        out[k] = torch.as_tensor(np.ascontiguousarray(v), dtype=dt, device=device)
    return out


def take(T, idx):
    return {k: v[idx] for k, v in T.items()}


def masked_mse(p, y, m):
    return ((p - y) ** 2 * m).sum() / m.sum().clamp(min=1)


def fit(model, T, tr, va, lr=1e-3, bs=256, epochs=40, patience=8, decay_every=10, kd=0.0, kd_id=0.0,
        seed=0, ema=0.99, min_epochs=10, log=print):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    avg = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best, best_state, best_epoch, bad = math.inf, None, 0, 0
    t0 = time.time()
    tr = torch.as_tensor(tr)
    for epoch in range(epochs):
        for group in opt.param_groups:
            group["lr"] = lr * 0.5 ** (epoch // decay_every)
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g)]
        tot = 0.0
        for idx in perm.split(bs):
            b = take(T, idx)
            q = q_id = None
            if kd > 0:
                q = (torch.rand(len(idx), generator=g) >= kd).float()
            if kd_id > 0:
                q_id = (torch.rand(len(idx), generator=g) >= kd_id).float()
            loss = masked_mse(model(b, q, q_id), b["yf"], b["ymask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                for k, val in model.state_dict().items():
                    if val.dtype.is_floating_point:
                        avg[k].mul_(ema).add_(val, alpha=1 - ema)
                    else:
                        avg[k].copy_(val)
            tot += loss.item() * len(idx)
        live = copy.deepcopy(model.state_dict())
        model.load_state_dict(avg)
        v = evaluate_mse(model, T, va)
        model.load_state_dict(live)
        log(f"epoch {epoch + 1}: train {tot / len(tr):.4f}  val {v:.4f}")
        if v < best:
            best, best_epoch, bad = v, epoch + 1, 0
            best_state = copy.deepcopy(avg)
        else:
            bad += 1
            if bad >= patience and epoch + 1 >= min_epochs:
                break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "val_mse": best, "train_time": time.time() - t0}


@torch.no_grad()
def predict(model, T, idx, q=None, q_id=None, bs=2048):
    model.eval()
    ys, xs = [], []
    idx = torch.as_tensor(idx)
    part = lambda a, j, n: None if a is None else torch.as_tensor(a[j * bs: j * bs + n], dtype=torch.float32)
    for j, sub in enumerate(idx.split(bs)):
        b = take(T, sub)
        y, xt, _ = model(b, part(q, j, len(sub)), part(q_id, j, len(sub)), return_parts=True)
        ys.append(y)
        xs.append(xt)
    return torch.cat(ys).numpy(), torch.cat(xs)


def evaluate_mse(model, T, idx, q=None, q_id=None):
    p, _ = predict(model, T, idx, q, q_id)
    y, m = T["yf"][idx].numpy(), T["ymask"][idx].numpy()
    return float(((p - y) ** 2 * m).sum() / max(m.sum(), 1))


@torch.no_grad()
def gate_statistics(model, T, idx, hours):
    """D2-ready statistics of the look-back (gate.py) with the knowledge trusted."""
    _, xt = predict(model, T, idx)
    i = torch.as_tensor(idx)
    return window_stats(xt, T["hodp"][i], hours, T["x"][i], T["xprof_l"][i], T["xprof_s"][i]).numpy()


def error_gate(p1, p0, y, m, node, day, warm=None, half_life=7.0):
    """Output-space combination f0 + w (f1 - f0) with w from the days before each forecast day
    (see the module docstring). warm: (p1, p0, y, m, node, day) of earlier days (validation)."""
    lam = 0.5 ** (1.0 / half_life)
    e = (y - p0) * m
    d = (p1 - p0) * m
    a_day, b_day = (e * d).sum(1), (d * d).sum(1)
    w = np.zeros(len(p1))
    for i in np.unique(node):
        A = B = 0.0
        last = None
        rows = []
        if warm is not None:
            wp1, wp0, wy, wm, wn, wd = warm
            sel = np.flatnonzero(wn == i)
            we, wdd = (wy[sel] - wp0[sel]) * wm[sel], (wp1[sel] - wp0[sel]) * wm[sel]
            rows += [(wd[r], (we[k] * wdd[k]).sum(), (wdd[k] ** 2).sum(), None) for k, r in enumerate(sel)]
        sel = np.flatnonzero(node == i)
        rows += [(day[r], a_day[r], b_day[r], r) for r in sel]
        rows.sort(key=lambda t: t[0])
        for dd, a, b, r in rows:
            if last is not None:
                f = lam ** (dd - last)
                A, B = A * f, B * f
            if r is not None:
                w[r] = float(np.clip(A / B, 0.0, 1.0)) if B > 1e-9 else 1.0
            A, B, last = A + a, B + b, dd
    return p0 + w[:, None] * (p1 - p0), w


def gate_modes(model, T, ref_idx, val_idx, test_idx, hours, oracle_q=None, has_id=True):
    """Test predictions for every evaluation mode (dict mode -> (n_test, H) array) and info."""
    node = T["node"].numpy()
    day = T["day"].numpy() if "day" in T else None
    y = T["yf"].numpy()
    m = T["ymask"].numpy()
    preds = {"trust": predict(model, T, test_idx)[0]}
    info = {}
    if model.uses_knowledge:
        preds["null"] = predict(model, T, test_idx, q=np.zeros(len(test_idx)))[0]
    if not has_id:
        return preds, info
    p0 = predict(model, T, test_idx, q_id=np.zeros(len(test_idx)))[0]
    preds["noid"] = p0
    ref = NodeReference().fit(gate_statistics(model, T, ref_idx, hours), node[ref_idx])
    d2_val = ref.d2(gate_statistics(model, T, val_idx, hours), node[val_idx])
    d2_test = ref.d2(gate_statistics(model, T, test_idx, hours), node[test_idx])
    preds["gate"] = predict(model, T, test_idx, q_id=posterior_q(d2_test, chi2_threshold(0.999)))[0]
    cands = {"none": None}
    cands.update({str(p): chi2_threshold(p) for p in GATE_PS})
    val_mse = {name: evaluate_mse(model, T, val_idx, q_id=None if c is None else posterior_q(d2_val, c))
               for name, c in cands.items()}
    choice = min(val_mse, key=val_mse.get)
    c = cands[choice]
    preds["gate_val"] = preds["trust"] if c is None else predict(
        model, T, test_idx, q_id=posterior_q(d2_test, c))[0]
    v1 = predict(model, T, val_idx)[0]
    v0 = predict(model, T, val_idx, q_id=np.zeros(len(val_idx)))[0]
    warm = (v1, v0, y[val_idx], m[val_idx], node[val_idx], day[val_idx])
    preds["gate_err"], w = error_gate(preds["trust"], p0, y[test_idx], m[test_idx], node[test_idx],
                                      day[test_idx], warm)
    q_lf = posterior_q(d2_test, chi2_threshold(0.999))
    preds["gate_comb"] = p0 + (w * q_lf)[:, None] * (preds["trust"] - p0)
    if oracle_q is not None:
        preds["oracle"] = predict(model, T, test_idx, q_id=oracle_q)[0]
    info = {"gate_val_choice": choice, "val_mse_by_threshold": val_mse,
            "test_gated_share": float((d2_test > chi2_threshold(0.999)).mean()),
            "gate_err_mean_w": float(w.mean())}
    return preds, info


def metrics(pred, S, idx, masks, hour_mask=None):
    """Per-mask metrics over (sample, hour) pairs: MSE / MAE in units of sigma and MAE in the
    original unit. masks: name -> boolean array over idx; hour_mask: name -> (len(idx), H)."""
    y, m = S["yf"][idx], S["ymask"][idx].astype(bool)
    raw = S["yf_raw"][idx]
    pr = pred * S["sigma"][idx][:, None]
    out = {}
    for name, mk in masks.items():
        mm = m & mk[:, None]
        if hour_mask is not None and name in hour_mask:
            mm = mm & hour_mask[name]
        n = int(mm.sum())
        if n == 0:
            continue
        e = (pred - y)[mm]
        out[name] = {"mse": float((e ** 2).mean()), "mae": float(np.abs(e).mean()),
                     "mae_kw": float(np.abs((pr - raw)[mm]).mean()), "n": n}
    return out
