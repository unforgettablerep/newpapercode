"""Samples (node, forecast day) for the RNK experiments.

A dataset is a dict of numpy arrays (hour index 0 is 00:00 of day 0):
  y       (N, T)     target in original units (NaN = missing)
  cov     (N, T, C)  covariates, known for past and future hours (weather is ex-post)
  cal     (T, Cc)    calendar features;  hod, dow (T,) hour of day and day of week
  cap0    (N, K)     capacity of each technology active at hour 0
  events  list of registry events (see registry.py)
  static  (N, S)     static descriptors
  kinds   names of the K technologies; pv_kind, irr_col: PV technology and irradiance column
  undoc_day (N,)     optional: day of an undocumented change (-1 = none; synthetic data only)

A sample is one node and one forecast day D: the look-back is hours 24 D - L .. 24 D - 1, the
target hours 24 D .. 24 D + H - 1. Everything a sample contains is computable at 00:00 of day D:
  sigma    load scale = mean |y| over the R days before D (the unit of the model; inductive)
  prof     (7 * 24,) mean of y / sigma by weekday and hour over those R days (near-term profile)
  prof_s   the same over the last R_short days (still in units of sigma): the short profile the
           fallback uses after a change (it adapts faster, but is noisier)
  desc     static descriptors + log(1 + registered capacity) + registered capacity / sigma
  capw     (L + H, K) capacities assumed for the look-back and target hours (registry.py scheme)
  xprof_l, xprof_s  the R-day and the short profile at the look-back hours (gate statistic)
Missing look-back values are filled with the profile; samples with > 20 % missing look-back or
> 50 % missing targets are dropped.
"""
import numpy as np
from .registry import build_posteriors, capacity_window, lag_prior


def rolling_scale_profile(y, dow, R, sigma=None):
    """sigma (N, days) and profile (N, days, 7, 24) from the R days before each day (NaN where
    fewer than 7 days are available). With `sigma` given, the profile is in its units."""
    N, T = y.shape
    days = T // 24
    yd = y[:, : days * 24].reshape(N, days, 24)
    ddow = dow[: days * 24 : 24]
    absd = np.nan_to_num(np.abs(yd)).sum(2)
    cntd = np.isfinite(yd).sum(2)
    c_abs = np.concatenate([np.zeros((N, 1)), np.cumsum(absd, 1)], 1)
    c_cnt = np.concatenate([np.zeros((N, 1)), np.cumsum(cntd, 1)], 1)
    onehot = np.eye(7)[ddow]                                          # (days, 7)
    val = np.nan_to_num(yd)[:, :, None, :] * onehot[None, :, :, None]  # (N, days, 7, 24)
    cnt = np.isfinite(yd)[:, :, None, :] * onehot[None, :, :, None]
    c_val = np.concatenate([np.zeros((N, 1, 7, 24)), np.cumsum(val, 1)], 1)
    c_c = np.concatenate([np.zeros((N, 1, 7, 24)), np.cumsum(cnt, 1)], 1)
    D = np.arange(days)
    lo = np.maximum(D - R, 0)
    if sigma is None:
        n_abs = c_cnt[:, D] - c_cnt[:, lo]
        sigma = (c_abs[:, D] - c_abs[:, lo]) / np.maximum(n_abs, 1)
        sigma[(D - lo)[None].repeat(N, 0) < 7] = np.nan
        sigma = np.maximum(sigma, 1e-3)
    s = c_val[:, D] - c_val[:, lo]
    n = c_c[:, D] - c_c[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = s / n / sigma[:, :, None, None]
    fill = np.nanmean(prof.reshape(N, days, -1), 2)                   # nodes' mean profile value
    prof = np.where(np.isfinite(prof), prof, np.nan_to_num(fill)[:, :, None, None])
    return sigma, prof


def build_samples(ds, days, nodes, L, H, R=28, scheme="registry", min_delta=0.0, prior_shape=2.0,
                  prior_scale=25.0, temper=0.5, R_short=14):
    """Arrays for all (node, day) pairs with node in `nodes` and day in `days`."""
    y, cov, cal, hod, dow = ds["y"], ds["cov"], ds["cal"], ds["hod"], ds["dow"]
    N, T = y.shape
    K = ds["cap0"].shape[1]
    sigma, prof = rolling_scale_profile(y, dow, R)
    _, prof_short = rolling_scale_profile(y, dow, R_short, sigma)
    events = ds["events"]
    by_node = {i: [] for i in range(N)}
    for n, e in enumerate(events):
        by_node[e["node"]].append((n, e))
    posts = None
    if scheme == "posterior":
        prior = lag_prior(shape=prior_shape, scale=prior_scale)
        sig_fn = lambda i, d: float(np.nan_to_num(sigma[i, min(d, sigma.shape[1] - 1)], nan=1.0))
        posts = build_posteriors(events, y, cov, hod, ds["irr_col"], ds["pv_kind"], sig_fn, prior,
                                 temper=temper)
    undoc = ds.get("undoc_day", np.full(N, -1))
    pairs = [(i, d) for i in nodes for d in days if 24 * d - L >= 0 and 24 * d + H <= T
             and np.isfinite(sigma[i, d])]
    node = np.array([p[0] for p in pairs])
    day = np.array([p[1] for p in pairs])
    t0 = 24 * day
    hist = t0[:, None] + np.arange(-L, 0)[None]
    fut = t0[:, None] + np.arange(H)[None]
    sig = sigma[node, day]
    x = y[node[:, None], hist] / sig[:, None]
    yf = y[node[:, None], fut] / sig[:, None]
    pr = prof[node, day]                                              # (S, 7, 24)
    xfill = pr[np.arange(len(pairs))[:, None], dow[hist], hod[hist]]
    ps = prof_short[node, day]
    xprof_s = ps[np.arange(len(pairs))[:, None], dow[hist], hod[hist]]
    miss_x = ~np.isfinite(x)
    x = np.where(miss_x, xfill, x)
    ymask = np.isfinite(yf)
    keep = (miss_x.mean(1) <= 0.2) & (ymask.mean(1) >= 0.5)

    reg = np.zeros((len(pairs), K))
    capw = np.zeros((len(pairs), L + H, K))
    reg_days = np.full(len(pairs), 9999)
    reg_kind = np.full(len(pairs), -1)
    reg_delta = np.zeros(len(pairs))
    reg_true = np.full(len(pairs), -1)                # 1: the latest known event is commissioned
    for s, (i, d) in enumerate(pairs):
        known = [(n, e) for n, e in by_node[i] if e["tau"] // 24 < d]
        reg[s] = ds["cap0"][i] + sum((np.eye(K)[e["kind"]] * e["delta"] for _, e in known),
                                     np.zeros(K))
        capw[s] = capacity_window(24 * d, L, H, ds["cap0"][i], by_node[i], scheme, posts,
                                  ds["pv_kind"], prior_shape, prior_scale)
        big = [e for _, e in known if abs(e["delta"]) >= min_delta]
        if big:
            e = max(big, key=lambda e: e["tau"])
            reg_days[s], reg_kind[s], reg_delta[s] = d - e["tau"] // 24, e["kind"], e["delta"]
            if "true" in e:
                reg_true[s] = int(e["true"] < 24 * (d + 1))
    desc = np.concatenate([ds["static"][node], np.log1p(np.maximum(reg, 0)),
                           reg / sig[:, None]], 1)
    out = {
        "x": x, "yf": np.nan_to_num(yf), "ymask": ymask, "sigma": sig,
        "yf_raw": np.nan_to_num(y[node[:, None], fut]),
        "covp": cov[node[:, None], hist], "covf": cov[node[:, None], fut],
        "calp": cal[hist], "calf": cal[fut], "hodp": hod[hist],
        "prof": pr.reshape(len(pairs), -1), "prof_s": ps.reshape(len(pairs), -1),
        "xprof_l": xfill, "xprof_s": xprof_s,
        "desc": desc, "capw": capw / sig[:, None, None],
        "node": node, "day": day, "reg_days": reg_days, "reg_kind": reg_kind,
        "reg_delta": reg_delta, "reg_true": reg_true,
        "undoc_days": np.where((undoc[node] >= 0) & (day >= undoc[node]), day - undoc[node], -1),
    }
    return {k: v[keep] for k, v in out.items()}
