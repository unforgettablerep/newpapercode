"""CPU tests of the RNK code: python -m pytest -q tests/test_rnk.py"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip("scipy")
from rnk.data import build_samples, rolling_scale_profile  # noqa: E402
from rnk.gate import chi2_threshold, posterior_q  # noqa: E402
from rnk.model import MLPBackbone, RNKForecaster  # noqa: E402
from rnk.registry import CommissioningPosterior, capacity_window, lag_prior  # noqa: E402
from rnk.sim import make_sim  # noqa: E402
from rnk.train import error_gate, to_torch  # noqa: E402


def test_capacity_schemes_and_registry_timing():
    e = {"node": 0, "kind": 0, "tau": 24 * 10, "delta": 5.0, "true": 24 * 20}
    ev = [(0, e)]
    # not known at origins up to the registry day itself
    for D in (5, 10):
        cap = capacity_window(24 * D, 48, 24, [1.0], ev, "registry")
        assert np.allclose(cap, 1.0)
    cap = capacity_window(24 * 12, 72, 24, [1.0], ev, "registry")
    hours = np.arange(24 * 12 - 72, 24 * 12 + 24)
    assert np.allclose(cap[:, 0], 1.0 + 5.0 * (hours >= 240))
    cap = capacity_window(24 * 12, 72, 24, [1.0], ev, "oracle")
    assert np.allclose(cap[:, 0], 1.0 + 5.0 * (hours >= 480))
    cap = capacity_window(24 * 12, 72, 24, [1.0], ev, "prior")
    assert np.all(np.diff(cap[:, 0]) >= -1e-12) and cap[0, 0] == 1.0 and cap[-1, 0] < 6.0


def test_posterior_without_data_is_the_prior_and_finds_a_change():
    prior = lag_prior()
    rng = np.random.default_rng(0)
    days, tau, change = 400, 200, 240
    g = rng.uniform(0.1, 0.8, days)
    z = 0.2 + 0.1 * g + rng.normal(0, 0.02, days)
    z[change:] -= 0.8 * 0.5 * g[change:]                 # delta / sigma = 0.5
    post = CommissioningPosterior(z, g, tau, 0.5, prior)
    c0 = post.cdf(tau, np.arange(tau, tau + 200))       # no post-registry data yet
    assert np.allclose(c0, np.cumsum(prior[:-1])[:200])
    c = post.cdf(change + 30, np.arange(tau, tau + 200))
    assert np.all(np.diff(c) >= -1e-12) and 0 <= c.min() and c.max() <= 1 + 1e-9
    assert c[change - tau - 5] < 0.1 and c[change - tau + 3] > 0.9
    assert post.cdf(change + 30, np.array([tau - 1]))[0] == 0.0


def test_profile_and_scale_use_only_the_past():
    rng = np.random.default_rng(1)
    y = rng.uniform(1, 2, (2, 24 * 60))
    dow = (np.arange(24 * 60) // 24) % 7
    s1, p1 = rolling_scale_profile(y, dow, 28)
    y2 = y.copy()
    y2[:, 24 * 40:] *= 10
    s2, p2 = rolling_scale_profile(y2, dow, 28)
    assert np.allclose(s1[:, :41], s2[:, :41], equal_nan=True)
    assert np.allclose(p1[:, :41], p2[:, :41], equal_nan=True)
    assert not np.allclose(s1[:, 41:], s2[:, 41:])


def small_sim():
    return make_sim(seed=0, n_seen=6, n_new=2, days=200, test_start=100)


def test_build_samples_shapes_and_inductive_fields():
    ds = small_sim()
    S = build_samples(ds, range(28, 200), range(8), 48, 24, scheme="posterior")
    n = len(S["x"])
    assert S["x"].shape == (n, 48) and S["yf"].shape == (n, 24) and S["capw"].shape == (n, 72, 2)
    assert S["prof"].shape == (n, 168) and S["prof_s"].shape == (n, 168)
    assert np.isfinite(S["x"]).all() and np.isfinite(S["desc"]).all()


def make_model(**kw):
    torch.manual_seed(0)
    return RNKForecaster(48, 24, 2, 5, 2, 8, 10, 168, hidden=32, d_k=8, **kw).eval()


def batch(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return {"x": r(n, 48), "covp": r(n, 48, 2), "covf": r(n, 24, 2), "calp": r(n, 48, 5),
            "calf": r(n, 24, 5), "capw": r(n, 72, 2).abs(), "prof": r(n, 168), "prof_s": r(n, 168),
            "desc": r(n, 10), "node": torch.arange(n), "id_known": torch.ones(n)}


def test_full_model_equals_the_backbone_at_initialisation():
    b = batch()
    base = make_model()
    full = make_model(use_id=True, use_desc=True, use_prof=True, use_basis=True, use_cap=True,
                      cap_signs=["mono", "free"])
    with torch.no_grad():
        assert torch.allclose(base(b), full(b), atol=1e-6)
        assert torch.allclose(full(b), full(b, q=torch.zeros(4), q_id=torch.zeros(4)), atol=1e-6)


def test_identity_gate_and_profile_mixing():
    full = make_model(use_id=True, use_prof=True)
    with torch.no_grad():
        full.know.ident.normal_()
    b = batch()
    k1 = full.knowledge(b)
    k0 = full.knowledge(b, q_id=torch.zeros(4))
    b2 = dict(b, prof=b["prof_s"])
    k_short_noid = full.knowledge(dict(b2, id_known=torch.zeros(4)))
    assert torch.allclose(k0, k_short_noid, atol=1e-6) and not torch.allclose(k0, k1)
    # unknown nodes have no identity
    bu = dict(b, id_known=torch.zeros(4))
    assert torch.allclose(full.knowledge(bu), full.knowledge(dict(bu, node=torch.zeros(4, dtype=torch.long))))


def test_error_gate_weights():
    rng = np.random.default_rng(0)
    n, H = 40, 24
    y = rng.normal(size=(n, H))
    m = np.ones((n, H))
    node, day = np.zeros(n, int), np.arange(n)
    f_good, f_bad = y + 0.01 * rng.normal(size=(n, H)), y + rng.normal(size=(n, H))
    _, w = error_gate(f_good, f_bad, y, m, node, day)
    assert w[10:].min() > 0.9
    _, w = error_gate(f_bad, f_good, y, m, node, day)
    assert w[10:].max() < 0.1


def test_chi2_threshold_false_alarm_rate():
    rng = np.random.default_rng(0)
    d2 = (rng.normal(size=(200000, 4)) ** 2).sum(1)
    c = chi2_threshold(0.99)
    assert abs((d2 > c).mean() - 0.01) < 0.002
    assert posterior_q(np.array([c]), c)[0] == pytest.approx(0.5)


def test_samples_to_torch():
    ds = small_sim()
    S = build_samples(ds, range(28, 60), range(3), 48, 24)
    S["id_known"] = np.ones(len(S["x"]), np.float32)
    T = to_torch(S)
    assert T["node"].dtype == torch.long and T["x"].dtype == torch.float32
    model = RNKForecaster(48, 24, 2, 5, 2, 8, S["desc"].shape[1], hidden=32, d_k=8, use_id=True,
                          use_desc=True, use_prof=True, use_basis=True, use_cap=True)
    out = model({k: v[:5] for k, v in T.items()})
    assert out.shape == (5, 24) and torch.isfinite(out).all()
    assert isinstance(model.backbone, MLPBackbone)
