"""One RNK experiment: build samples, train one variant, evaluate every gate mode on the
protocol masks (newpaper papers/model_direction.md, 5.2).

Variants (all on the same global MLP backbone):
  global       no node knowledge (channel-independent global model)
  id           + zero-initialised node identity (transductive; "always trust")
  id_drop      + identity with identity dropout 0.2 (trains the fallback; gates evaluated)
  nke_noid     + descriptors and near-term profile (NKE), no identity
  nke          NKE + identity
  kcd_reg      NKE + covariate-consistent decomposition, registry date = commissioning date
  kcd_prior    the same with the lag prior
  kcd_post     the same with the commissioning posterior
  kcd_oracle   the same with the true commissioning dates (synthetic data only)
  rnk          kcd_post + persistence dropout 0.2 (identity off, short profile): the full
               model (gates evaluated)
  rnk_noid     kcd_post without the identity
  rnk_all      kcd_post + dropout of the whole knowledge 0.2 (ablation: gating everything)
Every model is evaluated in each mode of train.gate_modes that applies to it.

Data
  sim       rnk/sim.py; look-back 48 h, horizon 24 h; train days 28-329, validation 330-359, test
            360-539; training nodes 0-79, new nodes 80-119
  feederbw  rnk/feederbw.py; look-back 168 h, horizon 24 h; day 0 = 2023-04-01; train days
            28-334 (to 2024-02-29), validation 335-365 (March 2024), test 366-730
            (2024-04-01 .. 2025-03-31); split "temporal" (all feeders) or "locational" (train on
            the first half of the feeders, test on the other half)
"""
import os
import pickle
import time

import numpy as np
import torch

from .data import build_samples
from .model import RNKForecaster
from .train import fit, gate_modes, metrics, to_torch

BASE = {"use_id": False, "use_desc": False, "use_prof": False, "use_basis": False, "use_cap": False,
        "scheme": "registry", "kd": 0.0, "kd_id": 0.0}
NKE = {"use_id": True, "use_desc": True, "use_prof": True}
KCD = dict(NKE, use_basis=True, use_cap=True)
VARIANTS = {
    "global": {},
    "id": {"use_id": True},
    "id_drop": {"use_id": True, "kd_id": 0.2},
    "nke_noid": {"use_desc": True, "use_prof": True},
    "nke": NKE,
    "kcd_reg": dict(KCD, scheme="registry"),
    "kcd_prior": dict(KCD, scheme="prior"),
    "kcd_post": dict(KCD, scheme="posterior"),
    "kcd_oracle": dict(KCD, scheme="oracle"),
    "rnk": dict(KCD, scheme="posterior", kd_id=0.2),
    "rnk_noid": dict(KCD, use_id=False, scheme="posterior"),
    "rnk_all": dict(KCD, scheme="posterior", kd=0.2),
}
DATA = {
    "sim": {"L": 48, "H": 24, "R": 28, "train": (28, 330), "val": (330, 360), "test": (360, 540),
            "gate_hours": 48},
    "feederbw": {"L": 168, "H": 24, "R": 28, "train": (28, 335), "val": (335, 366), "test": (366, 731),
                 "gate_hours": 72},
}


def load_dataset(cfg):
    if cfg["data"] == "sim":
        from .sim import make_sim
        ds = make_sim(seed=cfg.get("sim_seed", 0))
        ds["signs"] = ("mono", "mono")
        return ds
    from .feederbw import load_hourly, make_feederbw
    hourly = load_hourly(cfg["zips"], os.path.join(cfg["cache"], "feederbw_hourly.pkl"))
    return make_feederbw(hourly, cfg["weather"], cfg["meta"])


def samples(cfg, ds, scheme):
    d = DATA[cfg["data"]]
    key = f"{cfg['data']}_{cfg.get('sim_seed', 0)}_{scheme}.pkl"
    path = os.path.join(cfg["cache"], key) if cfg.get("cache") else None
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    N = ds["y"].shape[0]
    S = build_samples(ds, range(d["train"][0], d["test"][1]), range(N), d["L"], d["H"], d["R"],
                      scheme=scheme, min_delta=10.0 if cfg["data"] == "feederbw" else 0.0)
    if path:
        os.makedirs(cfg["cache"], exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(S, fh)
    return S


def split_masks(cfg, ds, S):
    d = DATA[cfg["data"]]
    day, node = S["day"], S["node"]
    period = lambda a, b: (day >= a) & (day < b)
    tr, va, te = period(*d["train"]), period(*d["val"]), period(*d["test"])
    N = ds["y"].shape[0]
    if cfg["data"] == "sim":
        seen_nodes = ds["seen"]
    elif cfg.get("split", "temporal") == "locational":
        seen_nodes = np.arange(N) < N // 2
    else:
        seen_nodes = np.ones(N, bool)
    seen = seen_nodes[node]
    undoc = S["undoc_days"]
    rd = S["reg_days"]
    pv = S["reg_kind"] == ds["pv_kind"]
    nod = undoc < 0
    m = {}
    for name, grp in (("seen", seen), ("new", ~seen)):
        if not grp.any():
            continue
        base = te & grp
        m[f"{name}_all"] = base
        m[f"{name}_sunny"] = base
        m[f"{name}_stable"] = base & nod & (rd > 150)
        for a, b in ((0, 30), (30, 60), (60, 120)):
            m[f"{name}_pv_{a}_{b}"] = base & nod & pv & (rd >= a) & (rd < b)
            m[f"{name}_pv_{a}_{b}_sunny"] = m[f"{name}_pv_{a}_{b}"]
        if (undoc >= 0).any():
            m[f"{name}_undoc_0_60"] = base & (undoc >= 0) & (undoc < 60)
    return tr & seen, va & seen, te, seen, m


def run(cfg, log=print):
    t_start = time.time()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 4))
    v = dict(BASE, **VARIANTS[cfg["variant"]])
    d = DATA[cfg["data"]]
    ds = load_dataset(cfg)
    S = samples(cfg, ds, v["scheme"])
    tr_m, va_m, te_m, seen, masks = split_masks(cfg, ds, S)
    S["id_known"] = seen.astype(np.float32)
    mu = S["desc"][tr_m].mean(0)
    sd = S["desc"][tr_m].std(0) + 1e-6
    S["desc"] = ((S["desc"] - mu) / sd).astype(np.float32)
    T = to_torch(S)
    tr, va, te = (np.flatnonzero(x) for x in (tr_m, va_m, te_m))
    ref = np.flatnonzero((S["day"] >= d["train"][0]) & (S["day"] < d["train"][1]))   # all nodes

    K = S["capw"].shape[2]
    model = RNKForecaster(d["L"], d["H"], S["covp"].shape[2], S["calp"].shape[2], K,
                          ds["y"].shape[0], S["desc"].shape[1], S["prof"].shape[1],
                          hidden=cfg.get("hidden", 512), use_id=v["use_id"], use_desc=v["use_desc"],
                          use_prof=v["use_prof"], use_basis=v["use_basis"], use_cap=v["use_cap"],
                          cap_signs=ds.get("signs"), dropout=cfg.get("dropout", 0.1))
    info = fit(model, T, tr, va, lr=cfg.get("lr", 1e-3), bs=cfg.get("batch_size", 256),
               epochs=cfg.get("epochs", 40), patience=cfg.get("patience", 8), kd=v["kd"],
               kd_id=v["kd_id"], seed=cfg["seed"], log=log)

    oracle = None
    if cfg["data"] == "sim":
        oracle = (S["undoc_days"][te] < 0).astype(float)
    preds, ginfo = gate_modes(model, T, ref, va, te, d["gate_hours"], oracle,
                              has_id=v["use_id"] or v["use_prof"])
    H = d["H"]
    irr = S["covf"][te][..., ds["irr_col"]]
    hours = np.arange(H) % 24
    sunny = (irr > 0.4) & np.isin(hours, np.arange(10, 16))[None]
    sub = {k: mk[te] for k, mk in masks.items()}
    hour_mask = {k: sunny for k in sub if k.endswith("_sunny")}
    res = {mode: metrics(p, S, te, sub, hour_mask) for mode, p in preds.items()}
    return {
        "cfg": cfg, "variant_cfg": v, "fit": info, "gate": ginfo, "metrics": res,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "time": time.time() - t_start,
    }
