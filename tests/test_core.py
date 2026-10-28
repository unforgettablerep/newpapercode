"""CPU tests: python -m pytest -q tests"""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idembed.data import borders, load_splits, time_features  # noqa: E402
from idembed.identity import VariableIdentity  # noqa: E402
from idembed.mamba import MambaTorch  # noqa: E402
from idembed.models import VariateForecaster  # noqa: E402


def small_model(backbone, identity="none", **kw):
    torch.manual_seed(0)
    return VariateForecaster(backbone, n_vars=5, seq_len=24, pred_len=8, d_model=32, d_ff=32,
                             e_layers=2, n_heads=4, d_state=4, identity=identity,
                             mamba_impl="torch", **kw)


@pytest.mark.parametrize("backbone", ["smamba", "itransformer"])
@pytest.mark.parametrize("identity,loc", [("add", "input"), ("lowrank", "input"), ("add", "every")])
def test_identity_starts_as_the_original_model(backbone, identity, loc):
    base, ident = small_model(backbone).eval(), small_model(backbone, identity, id_loc=loc).eval()
    x, m = torch.randn(3, 24, 5), torch.randn(3, 24, 4)
    with torch.no_grad():
        assert torch.equal(base(x, m), ident(x, m))


@pytest.mark.parametrize("backbone", ["smamba", "itransformer"])
def test_shapes_with_and_without_marks_and_subsets(backbone):
    model = small_model(backbone, "add")
    x = torch.randn(2, 24, 5)
    assert model(x).shape == (2, 8, 5)
    assert model(x, torch.randn(2, 24, 4)).shape == (2, 8, 5)
    idx = torch.tensor([0, 3])
    assert model(x[..., idx], None, idx).shape == (2, 8, 2)


def test_identity_changes_output_once_trained():
    model = small_model("itransformer", "add")
    with torch.no_grad():
        model.identity[0].table.normal_()
    x = torch.randn(2, 24, 5)
    base = small_model("itransformer").eval()
    with torch.no_grad():
        assert not torch.allclose(model.eval()(x), base(x))


def test_identity_dropout_only_in_training():
    ident = VariableIdentity(4, 8, "add", p_drop=1.0)
    with torch.no_grad():
        ident.table.fill_(1.0)
    h = torch.zeros(2, 6, 8)
    assert ident.train()(h).abs().sum() == 0                      # everything dropped
    out = ident.eval()(h)
    assert torch.all(out[:, :4] == 1) and torch.all(out[:, 4:] == 0)  # calendar tokens untouched


def test_unseen_rows_get_no_gradient_and_can_be_switched_off():
    for mode in ("add", "lowrank"):
        ident = VariableIdentity(6, 4, mode, rank=2)
        if mode == "lowrank":
            with torch.no_grad():
                ident.V.normal_()
        seen = torch.tensor([0, 2, 5])
        ident(torch.zeros(1, 3, 4), seen).sum().backward()
        if mode == "add":
            g = ident.table.grad
            assert g[seen].abs().sum() > 0 and g[[1, 3, 4]].abs().sum() == 0
        mask = torch.zeros(6, dtype=torch.bool)
        mask[seen] = True
        ident.active.copy_(mask)
        with torch.no_grad():
            out = ident.eval()(torch.zeros(1, 6, 4))
        assert out[0, [1, 3, 4]].abs().sum() == 0


@pytest.mark.parametrize("chunk", [3, 4, 7])
def test_torch_mamba_scan_matches_sequential_recurrence(chunk):
    torch.manual_seed(0)
    m = MambaTorch(16, d_state=4, chunk=chunk)
    b, L, d, s = 2, 19, 16, 4
    dA, u, C = -torch.rand(b, L, d, s) * 3, torch.randn(b, L, d, s), torch.randn(b, L, s)
    h, ref = torch.zeros(b, d, s), []
    for t in range(L):
        h = torch.exp(dA[:, t]) * h + u[:, t]
        ref.append(torch.einsum("bds,bs->bd", h, C[:, t]))
    assert torch.allclose(m.scan(dA, u, C), torch.stack(ref, 1), atol=1e-4)


def test_torch_mamba_is_causal():
    torch.manual_seed(0)
    m = MambaTorch(16, d_state=4).eval()
    x = torch.randn(1, 12, 16)
    x2 = x.clone()
    x2[:, 7] += 1.0
    with torch.no_grad():
        d = (m(x2) - m(x)).abs().amax(-1)[0]
    assert torch.all(d[:7] == 0) and d[7] > 0


def test_official_split_borders():
    assert borders("ett_hour", 17420, 96) == ([0, 8544, 11424], [8640, 11520, 14400])
    n = 26304
    tr, te = int(n * 0.7), int(n * 0.2)
    assert borders("custom", n, 96) == ([0, tr - 96, n - te - 96], [tr, n - te, n])


def test_time_features_range():
    f = time_features(pd.date_range("2020-01-01", periods=50, freq="15min"), "t")
    assert f.shape == (50, 5) and f.min() >= -0.5 and f.max() <= 0.5


def write_fake_data(root):
    rng = np.random.default_rng(0)
    n = 2000
    t = np.arange(n)
    base = np.sin(2 * np.pi * t / 24)
    cols = {f"v{i}": base * (i + 1) + 0.1 * rng.standard_normal(n) for i in range(5)}
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="h"), **cols,
                       "OT": base + 0.1 * rng.standard_normal(n)})
    (root / "weather").mkdir(parents=True)
    df[["date", "OT"] + list(cols)].to_csv(root / "weather" / "weather.csv", index=False)
    (root / "PEMS").mkdir()
    np.savez(root / "PEMS" / "PEMS08.npz", data=rng.standard_normal((1500, 7, 3)).astype(np.float32))


def test_custom_loader_moves_target_last_and_scales(tmp_path):
    write_fake_data(tmp_path)
    (train, val, test), cols = load_splits("Weather", tmp_path, 24, 8)
    assert cols[-1] == "OT" and len(cols) == 6
    assert train.marks.shape[1] == 4
    assert abs(float(train.values.mean())) < 1e-4
    x, m, y = train.batch(torch.tensor([0, 5]))
    assert x.shape == (2, 24, 6) and m.shape == (2, 24, 4) and y.shape == (2, 8, 6)
    (ptrain, _, _), pcols = load_splits("PEMS08", tmp_path, 24, 8)
    assert ptrain.marks is None and len(pcols) == 7


def test_end_to_end_cli(tmp_path):
    write_fake_data(tmp_path)
    out = tmp_path / "res.jsonl"
    common = [sys.executable, str(ROOT / "run.py"), "--data", "Weather", "--root-path", str(tmp_path),
              "--pred-len", "8", "--seq-len", "24", "--d-model", "16", "--d-ff", "16",
              "--e-layers", "1", "--epochs", "2", "--batch-size", "64", "--device", "cpu",
              "--out", str(out)]
    for args in (["--model", "smamba", "--identity", "add", "--id-lr-mult", "10",
                  "--save-ckpt", str(tmp_path / "ck")],
                 ["--model", "itransformer", "--identity", "lowrank", "--id-rank", "4"],
                 ["--model", "smamba", "--identity", "add", "--id-dropout", "0.3",
                  "--train-var-ratio", "0.5"]):
        subprocess.run(common + args, check=True, cwd=ROOT, capture_output=True)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 3 and all(np.isfinite(r["mse"]) for r in rows)
    assert "mse_unseen" in rows[2] and "mse_seen" in rows[2]
    summary = subprocess.run([sys.executable, str(ROOT / "tools" / "summarize.py"), str(out)],
                             check=True, cwd=ROOT, capture_output=True, text=True).stdout
    assert "Weather" in summary
    ck = next((tmp_path / "ck").glob("*.pt"))
    res = subprocess.run([sys.executable, str(ROOT / "tools" / "analyze_identity.py"), str(ck),
                          "--root-path", str(tmp_path)], check=True, cwd=ROOT,
                         capture_output=True, text=True).stdout
    assert "Spearman" in res
