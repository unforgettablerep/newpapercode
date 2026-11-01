"""Synthetic multi-node load with known structure (newpaper papers/model_direction.md, 5).

Hourly net load of N nodes (feeders) over `days` days, in kW:

  y = size * base(type, weekday, hour) * weekly_i(weekday, hour) * season
      + HP capacity * heating(temperature, hour)
      - active PV capacity * 0.8 * irradiance (kW/m2)
      + AR(1) noise

- type: residential / commercial daily shapes (shared).
- weekly_i: node-specific weekend factor and a node-specific 4-hour bump on one weekday. With a
  2-day look-back this structure is mostly invisible in the window (the identity gap,
  theory.md section 2); the near-term profile and the identity can recover it.
- Registered PV additions: delta kW registered on day tau, commissioned after a lag drawn from
  Gamma(2, 25 days) (mean 50 days); the registry only knows tau.
- Undocumented changes ("load transfer"): from a day in the test period, the base becomes a
  mixture with another node type's shape and a level factor. Not in the registry.
- Covariates per region: irradiance (clear sky x clouds) and temperature; calendar features.

Nodes 0 .. n_seen - 1 are used for training; the others are new nodes (never trained on, but
their history is available for profiles and gate references).
"""
import numpy as np
from scipy.signal import lfilter

KINDS = ("pv", "hp")


def daily_shapes():
    h = np.arange(24)
    bump = lambda c, w: np.exp(-0.5 * ((h - c) / w) ** 2)
    res_wd = 0.55 + 0.35 * bump(7.5, 1.5) + 0.7 * bump(19, 2.0)
    res_we = 0.6 + 0.35 * bump(11, 3.0) + 0.6 * bump(19, 2.0)
    com_wd = 0.35 + 0.9 / (1 + np.exp(-(h - 7.5) * 1.5)) / (1 + np.exp((h - 18) * 1.5))
    com_we = 0.35 + 0.1 * bump(12, 4.0)
    norm = lambda a: a / a.mean()
    return {0: (norm(res_wd), norm(res_we)), 1: (norm(com_wd), norm(com_we))}


def make_sim(seed=0, n_seen=80, n_new=40, days=540, test_start=360, n_regions=4,
             undoc_share=0.2, pv_add_rate=0.8, lag_shape=2.0, lag_scale=25.0):
    rng = np.random.default_rng(seed)
    N, T = n_seen + n_new, days * 24
    t = np.arange(T)
    hod, day = t % 24, t // 24
    dow = day % 7
    doy = day % 365
    weekend = dow >= 5

    # covariates per region
    clear = np.clip(np.sin(np.pi * (hod - 6) / 12), 0, None) * (0.65 + 0.35 * np.sin(2 * np.pi * (doy - 80) / 365))
    temp_season = 10 - 9 * np.cos(2 * np.pi * (doy - 15) / 365) + 4 * np.sin(2 * np.pi * (hod - 9) / 24)
    irr_r, temp_r = np.zeros((n_regions, T)), np.zeros((n_regions, T))
    for r in range(n_regions):
        c = np.zeros(days)
        an = np.zeros(days)
        for d in range(1, days):
            c[d] = 0.7 * c[d - 1] + rng.normal(0, 0.5)
            an[d] = 0.8 * an[d - 1] + rng.normal(0, 1.5)
        cloud = 1 / (1 + np.exp(-(c[day] + rng.normal(0, 0.3, T) - 0.3)))
        irr_r[r] = np.clip(clear * (1 - 0.8 * cloud), 0, None)            # kW/m2
        temp_r[r] = temp_season + an[day]
    region = rng.integers(0, n_regions, N)

    shapes = daily_shapes()
    ntype = (rng.random(N) < 0.3).astype(int)                             # 1 = commercial
    size = np.exp(rng.normal(np.log(40), 0.5, N))
    wkend_f = rng.uniform(0.7, 1.3, N)
    bump_day = rng.integers(0, 7, N)
    bump_h = rng.integers(8, 17, N)
    bump_a = rng.uniform(0.0, 0.5, N)
    hp_cap = np.where(rng.random(N) < 0.3, rng.uniform(0.1, 0.5, N) * size, 0.0)
    pv0 = np.where(rng.random(N) < 0.5, rng.uniform(0.2, 1.0, N) * size, 0.0)

    events = []
    for i in range(N):
        for _ in range(rng.poisson(pv_add_rate)):
            tau_d = int(rng.integers(60, days - 60))
            lag = rng.gamma(lag_shape, lag_scale)
            events.append({"node": i, "kind": 0, "tau": 24 * tau_d, "delta": float(rng.uniform(0.3, 1.2) * size[i]),
                           "true": int(24 * tau_d + round(lag * 24))})
    undoc_day = np.full(N, -1)
    pick = rng.random(N) < undoc_share
    undoc_day[pick] = rng.integers(test_start + 20, days - 60, pick.sum())
    mix = rng.uniform(0.4, 0.7, N)
    lvl = np.where(rng.random(N) < 0.5, rng.uniform(1.3, 1.6, N), rng.uniform(0.6, 0.8, N))

    y = np.zeros((N, T))
    season = 1 + 0.08 * np.cos(2 * np.pi * (doy - 15) / 365)
    for i in range(N):
        wd, we = shapes[ntype[i]]
        base = np.where(weekend, we[hod], wd[hod]) * np.where(weekend, wkend_f[i], 1.0)
        base = base + bump_a[i] * ((dow == bump_day[i]) & (hod >= bump_h[i]) & (hod < bump_h[i] + 4))
        if undoc_day[i] >= 0:
            owd, owe = shapes[1 - ntype[i]]
            other = np.where(weekend, owe[hod], owd[hod])
            after = day >= undoc_day[i]
            base = np.where(after, lvl[i] * ((1 - mix[i]) * base + mix[i] * other), base)
        irr, temp = irr_r[region[i]], temp_r[region[i]]
        heat = np.clip((16 - temp) / 20, 0, 1) * (0.8 + 0.2 * np.cos(2 * np.pi * (hod - 7) / 24))
        pv = np.full(T, pv0[i])
        for e in events:
            if e["node"] == i:
                pv = pv + e["delta"] * (t >= e["true"])
        noise = lfilter([1.0], [1.0, -0.7], rng.normal(0, 0.06, T))
        y[i] = size[i] * base * season * (1 + noise) + hp_cap[i] * heat - pv * 0.8 * irr

    cov = np.stack([irr_r[region], (temp_r[region] - 10) / 10], -1)       # (N, T, 2)
    cal = np.stack([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
                    np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), weekend.astype(float)], 1)
    static = np.stack([np.log(size * rng.lognormal(0, 0.1, N)), ntype.astype(float)]
                      + [(region == r).astype(float) for r in range(n_regions)], 1)
    return {
        "y": y, "cov": cov, "cal": cal, "hod": hod, "dow": dow,
        "cap0": np.stack([pv0, hp_cap], 1), "events": events, "static": static,
        "kinds": KINDS, "pv_kind": 0, "irr_col": 0, "undoc_day": undoc_day,
        "seen": np.arange(N) < n_seen, "days": days, "test_start": test_start,
        "truth": {"bump_a": bump_a, "ntype": ntype, "size": size},
    }
