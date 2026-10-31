"""Training and evaluation.

Protocol (as in the official long-term forecasting experiments): Adam, MSE loss, learning
rate halved after every epoch from the second on ("type1"), early stopping on validation MSE
with patience 3, the best-validation weights are evaluated on every test window. Metrics are
MSE / MAE on standardised values.

Unseen-variable protocol (`train_var_ratio` < 1): a fixed random subset of the variables is
used for training and validation; the test set is scored on all variables and separately on
the seen and the unseen ones. Unseen variables keep the zero identity.

With `eval_orders`, the validation MSE of the same averaging is stored as well
(`val_mse_orders[K]`; early stopping always uses the single original-order pass).

`drift_gate`: label-free test-time gates for the variable-specific parts (identity, channel-norm
offsets, cycle offsets); papers/theory.md (newpaper repository) section 6 derives them. Window
statistics s = (mean, log std) of a variable's look-back window are in training units.
- posterior (stored as `mse_gate`): D2 = Mahalanobis distance of s from the variable's own
  training-window statistics; persistence posterior w = sigmoid((c - D2) / 2); the forecast is
  w * (with the parts) + (1 - w) * (with the parts switched off where D2 > c), mixed in the
  output space. c is a chi-square(2) quantile chosen on the validation split.
- hard (`mse_gate_hard`): the parts are switched off where |mean| + |log std| exceeds a quantile
  of the training windows' scores, chosen on the validation split.
Both include "no gate" as a candidate. The test MSE of every candidate is stored under
result["gate"] for analysis only.

`id_lr_mult` / `vs_lr_mult` = "auto": the multiplier is max(1, 1 / B), where B = lr * S * (3 -
2^(2-E)) is the learning budget (largest possible Adam displacement) of a zero-initialised
parameter over the planned E epochs of S steps (theory section 5).

`order_probe` = R: S-Mamba is also tested in R single random scan orders (`mse_rand1`, mean;
each order's results under result["rand_orders"]).

`per_var`: also store, for every variable (in the file's column order), the test and
validation MSE and the level / scale of its validation and test splits in units of the
training statistics, under result["per_var"].
"""
import copy
import math
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from .data import load_splits, permutation
from .models import VariateForecaster


GATE_QS = (0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
CHI2_PS = (0.9, 0.95, 0.99, 0.995, 0.999)            # c = chi-square(2) quantile = -2 log(1 - p)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def drift_score(x):
    """x: (B, L, n) standardised look-back windows -> (B, n) drift scores."""
    sd = torch.sqrt(torch.var(x, dim=1, unbiased=False) + 1e-10)
    return x.mean(1).abs() + torch.log(sd).abs()


def window_stats(x):
    """x: (B, L, n) standardised windows -> (B, n, 2): mean and log std."""
    sd = torch.sqrt(torch.var(x, dim=1, unbiased=False) + 1e-10)
    return torch.stack([x.mean(1), torch.log(sd)], -1)


@torch.no_grad()
def variable_window_stats(split, n=4096):
    """Per-variable mean (N, 2) and inverse covariance (N, 2, 2) of the window statistics of
    (up to n) windows of `split`."""
    g = torch.Generator().manual_seed(0)
    starts = torch.randperm(split.n_windows, generator=g)[:n].to(split.values.device)
    s = window_stats(split.batch(starts)[0].float())            # (n, N, 2)
    mu = s.mean(0)
    d = s - mu
    cov = torch.einsum("bni,bnj->nij", d, d) / max(len(s) - 1, 1)
    cov = cov + 1e-4 * torch.eye(2, device=cov.device)
    return mu, torch.linalg.inv(cov)


def mahalanobis(x, mu, icov):
    d = window_stats(x.float()) - mu
    return torch.einsum("bni,nij,bnj->bn", d, icov, d)


@torch.no_grad()
def drift_thresholds(split, var_idx=None, qs=GATE_QS, n=4096):
    """Quantiles of the drift scores of (up to n) windows of `split`."""
    g = torch.Generator().manual_seed(0)
    starts = torch.randperm(split.n_windows, generator=g)[:n].to(split.values.device)
    s = drift_score(split.batch(starts, var_idx)[0].float()).flatten()
    s = s[:: max(1, s.numel() // 4_000_000)]                    # torch.quantile size limit
    return {q: torch.quantile(s, q).item() for q in qs}


@torch.no_grad()
def evaluate(model, split, var_idx=None, bs=256, amp_dtype=None, groups=None, per_var=False,
             gate=None):
    """Returns {'mse', 'mae'} over all windows; with `groups` ({name: column positions}) also
    per-group metrics; with `per_var` also 'mse_per_var' / 'mae_per_var' (lists). `gate(x)`
    returns (keep, w): keep (B, n) switches the variable-specific parts off where 0; with w
    (B, n) the forecast is w * (all parts on) + (1 - w) * (with keep). 'gated' is the fraction
    of (window, variable) pairs with keep = 0."""
    model.eval()
    dev = split.values.device
    se_var = ae_var = 0.0
    n_gated = n_pairs = 0
    sums = {"all": [0.0, 0.0, 0]}
    for g in groups or {}:
        sums[g] = [0.0, 0.0, 0]
    for starts in torch.arange(split.n_windows, device=dev).split(bs):
        x, m, y = split.batch(starts, var_idx)
        keep = w = None
        if gate is not None:
            keep, w = gate(x)
            n_gated, n_pairs = n_gated + int((keep == 0).sum()), n_pairs + keep.numel()
        with torch.autocast(device_type=dev.type, dtype=amp_dtype or torch.float32,
                            enabled=amp_dtype is not None):
            p = model(x, m, var_idx, split.phase(starts), keep)
            if w is not None:
                full = p if bool((keep == 1).all()) else model(x, m, var_idx, split.phase(starts))
                p = w[:, None] * full + (1 - w[:, None]) * p
        err = p.float() - y
        if per_var:
            se_var = se_var + (err ** 2).sum((0, 1))
            ae_var = ae_var + err.abs().sum((0, 1))
        for g, (se, ae, cnt) in list(sums.items()):
            e = err if g == "all" else err[..., groups[g]]
            sums[g] = [se + (e ** 2).sum().item(), ae + e.abs().sum().item(), cnt + e.numel()]
    out = {}
    for g, (se, ae, cnt) in sums.items():
        k = "" if g == "all" else f"_{g}"
        out[f"mse{k}"], out[f"mae{k}"] = se / cnt, ae / cnt
    if per_var:
        n = split.n_windows * split.pred_len
        out["mse_per_var"], out["mae_per_var"] = (se_var / n).tolist(), (ae_var / n).tolist()
    if gate is not None:
        out["gated"] = n_gated / max(n_pairs, 1)
    return out


def hard_gate(tau):
    def gate(x):
        return (drift_score(x) <= tau).to(x.dtype), None
    return gate


def posterior_gate(c, mu, icov):
    def gate(x):
        d2 = mahalanobis(x, mu, icov)
        return (d2 <= c).to(x.dtype), torch.sigmoid((c - d2) / 2).to(x.dtype)
    return gate


def drift_gate(model, train, val, test, amp_dtype, orders=1):
    """Choose each gate's threshold on the validation split (see the module docstring)."""
    model.set_eval_orders(orders)
    taus = drift_thresholds(train)
    mu, icov = variable_window_stats(train)
    candidates = {
        "hard": [("1.0", None)] + [(str(q), hard_gate(taus[q])) for q in GATE_QS],
        "posterior": [("1.0", None)] + [(str(p), posterior_gate(-2 * math.log(1 - p), mu, icov))
                                        for p in CHI2_PS],
    }
    res = {"orders": orders, "tau_hard": {str(q): t for q, t in taus.items()}}
    for mode, cands in candidates.items():
        r = {"val": {}, "test": {}, "mae_test": {}, "gated_test": {}}
        for name, gate in cands:                 # "1.0": no gate
            v = evaluate(model, val, None, amp_dtype=amp_dtype, gate=gate)
            t = evaluate(model, test, None, amp_dtype=amp_dtype, gate=gate)
            r["val"][name], r["test"][name] = v["mse"], t["mse"]
            r["mae_test"][name], r["gated_test"][name] = t["mae"], t.get("gated", 0.0)
        r["q"] = min(r["val"], key=r["val"].get)
        res[mode] = r
    model.set_eval_orders(1)
    return res


def budget(lr, steps, epochs):
    """Largest Adam displacement of a parameter over `epochs` epochs of the type1 schedule."""
    return lr * steps * (3 - 2 ** (2 - epochs))


def run(cfg, log=print):
    """cfg: dict with data/model/training keys (see run.py). Returns a result dict."""
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    if cfg.get("tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.get("amp", "none")]
    (train, val, test), cols = load_splits(cfg["data"], cfg["root_path"], cfg["seq_len"],
                                           cfg["pred_len"], cfg["use_marks"], device,
                                           cfg.get("var_perm"))
    n_vars = train.values.shape[1]

    seen = None
    if cfg.get("train_var_ratio", 1.0) < 1.0:
        g = torch.Generator().manual_seed(cfg["seed"])
        k = max(1, int(round(n_vars * cfg["train_var_ratio"])))
        seen = torch.randperm(n_vars, generator=g)[:k].sort().values.to(device)

    model = VariateForecaster(
        cfg["model"], n_vars, cfg["seq_len"], cfg["pred_len"], d_model=cfg["d_model"],
        d_ff=cfg["d_ff"], e_layers=cfg["e_layers"], n_heads=cfg["n_heads"], d_state=cfg["d_state"],
        dropout=cfg["dropout"], activation=cfg["activation"], use_norm=cfg["use_norm"],
        identity=cfg["identity"], id_rank=cfg["id_rank"], id_loc=cfg["id_loc"],
        id_dropout=cfg["id_dropout"], mamba_impl=cfg["mamba_impl"],
        scan_order=cfg.get("scan_order", "fixed"), cn=cfg.get("cn", False),
        rcf_cycle=cfg.get("rcf_cycle", 0), vs_dropout=cfg.get("vs_dropout", 0.0),
        rcf_shared=cfg.get("rcf_shared", False)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # identity parameters start at zero and Adam moves them by about lr per step, so with the
    # small official learning rates (5e-5 .. 1e-3) they can be given a larger one
    id_params = list(model.identity.parameters())
    vs_params = model.variable_specific_parameters()      # channel-norm offsets, cycles
    own = {id(p) for p in id_params + vs_params}
    B = budget(cfg["lr"], train.n_windows // cfg["batch_size"], cfg["epochs"])
    mult = lambda m: max(1.0, 1.0 / B) if m == "auto" else float(m)
    id_mult, vs_mult = mult(cfg.get("id_lr_mult", 1.0)), mult(cfg.get("vs_lr_mult", 1.0))
    groups = [{"params": [p for p in model.parameters() if id(p) not in own], "mult": 1.0}]
    if id_params:
        groups.append({"params": id_params, "mult": id_mult})
    if vs_params:
        groups.append({"params": vs_params, "mult": vs_mult})
    opt = torch.optim.Adam(groups, lr=cfg["lr"])
    use_scaler = amp_dtype == torch.float16 and device.type == "cuda"
    scaler = (torch.amp.GradScaler("cuda", enabled=use_scaler) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=use_scaler))         # PyTorch < 2.3

    bs = cfg["batch_size"]
    steps = train.n_windows // bs                              # drop_last for training
    best, best_state, best_epoch, bad = math.inf, None, 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        lr = cfg["lr"] * 0.5 ** max(0, epoch - 1)              # "type1" schedule
        for group in opt.param_groups:
            group["lr"] = lr * group["mult"]
        model.train()
        order = torch.randperm(train.n_windows, device=device)[: steps * bs].view(steps, bs)
        loss_sum = 0.0
        for starts in order:
            x, m, y = train.batch(starts, seen)
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None):
                loss = F.mse_loss(model(x, m, seen, train.phase(starts)).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()
        v = evaluate(model, val, seen, amp_dtype=amp_dtype)["mse"]
        log(f"epoch {epoch + 1}: train {loss_sum / max(steps, 1):.4f}  val {v:.4f}  lr {lr:.2e}")
        if v < best:
            best, best_epoch, bad = v, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    groups = None
    if seen is not None:
        mask = torch.zeros(n_vars, dtype=torch.bool, device=device)
        mask[seen] = True
        groups = {"seen": mask.nonzero().squeeze(1), "unseen": (~mask).nonzero().squeeze(1)}
        for ident in model.identity:              # unseen variables get the zero identity
            ident.active.copy_(mask)
    t1 = time.time()
    per_var = bool(cfg.get("per_var"))
    metrics = evaluate(model, test, None, amp_dtype=amp_dtype, groups=groups, per_var=per_var)
    ks = cfg.get("eval_orders", 1)
    ks = sorted(k for k in ([ks] if isinstance(ks, int) else ks) if k > 1)
    for k in ks:                                   # same model, scan-order averaging at test
        model.set_eval_orders(k)
        avg = evaluate(model, test, None, amp_dtype=amp_dtype, groups=groups)
        avg["val_mse"] = evaluate(model, val, seen, amp_dtype=amp_dtype)["mse"]   # for selection
        avg = {("val_mse_orders" if name == "val_mse" else f"{name}_orders"): v
               for name, v in avg.items()}
        if len(ks) > 1:
            metrics.update({f"{name}{k}": v for name, v in avg.items()})
        if k == ks[-1]:                            # '_orders': the largest number of orders
            metrics.update(avg)
        model.set_eval_orders(1)
    if cfg.get("drift_gate"):
        if seen is not None:
            raise ValueError("drift_gate is not supported with train_var_ratio < 1")
        gate = drift_gate(model, train, val, test, amp_dtype, ks[-1] if ks else 1)
        post, hard = gate["posterior"], gate["hard"]
        metrics.update(gate=gate, mse_gate=post["test"][post["q"]], mae_gate=post["mae_test"][post["q"]],
                       mse_gate_hard=hard["test"][hard["q"]], mae_gate_hard=hard["mae_test"][hard["q"]])
    if cfg.get("order_probe") and model.mamba_backend() is not None:
        probe = {"seeds": [], "test": [], "mae_test": [], "val": []}
        for r in range(cfg["order_probe"]):
            model.set_eval_seed(1000 + r)
            t = evaluate(model, test, None, amp_dtype=amp_dtype)
            probe["seeds"].append(1000 + r)
            probe["test"].append(t["mse"])
            probe["mae_test"].append(t["mae"])
            probe["val"].append(evaluate(model, val, seen, amp_dtype=amp_dtype)["mse"])
        model.set_eval_seed(None)
        metrics.update(rand_orders=probe, mse_rand1=float(np.mean(probe["test"])),
                       mae_rand1=float(np.mean(probe["mae_test"])))
    test_time = time.time() - t1
    if per_var:
        pv = {"test_mse": metrics.pop("mse_per_var"), "test_mae": metrics.pop("mae_per_var")}
        if seen is None:
            pv["val_mse"] = evaluate(model, val, None, amp_dtype=amp_dtype, per_var=True)["mse_per_var"]
        for name, split in (("val", val), ("test", test)):
            pv[f"{name}_mean"] = split.values.mean(0).tolist()
            pv[f"{name}_std"] = split.values.std(0, unbiased=False).tolist()
        inv = (np.argsort(permutation(n_vars, cfg["var_perm"])) if cfg.get("var_perm") is not None
               else np.arange(n_vars))             # back to the file's column order
        pv = {k: [float(f"{v[i]:.6g}") for i in inv] for k, v in pv.items()}
        pv["columns"] = [cols[i] for i in inv]
        metrics["per_var"] = pv
    result = dict(cfg, val_mse=best, best_epoch=best_epoch, epochs_run=epoch + 1,
                  train_time_s=round(train_time, 1), test_time_s=round(test_time, 1),
                  n_params=n_params, n_vars=n_vars, mamba_backend=model.mamba_backend(), **metrics)
    if "auto" in (cfg.get("id_lr_mult"), cfg.get("vs_lr_mult")):
        result.update(budget=B, id_lr_mult_used=id_mult, vs_lr_mult_used=vs_mult)
    if device.type == "cuda":
        result["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    return result, model, cols
