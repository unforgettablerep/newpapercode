"""Training and evaluation.

Protocol (as in the official long-term forecasting experiments): Adam, MSE loss, learning
rate halved after every epoch from the second on ("type1"), early stopping on validation MSE
with patience 3, the best-validation weights are evaluated on every test window. Metrics are
MSE / MAE on standardised values.

Unseen-variable protocol (`train_var_ratio` < 1): a fixed random subset of the variables is
used for training and validation; the test set is scored on all variables and separately on
the seen and the unseen ones. Unseen variables keep the zero identity.
"""
import copy
import math
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from .data import load_splits
from .models import VariateForecaster


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, split, var_idx=None, bs=256, amp_dtype=None, groups=None):
    """Returns {'mse', 'mae'} over all windows; with `groups` ({name: column positions}) also
    per-group metrics."""
    model.eval()
    dev = split.values.device
    sums = {"all": [0.0, 0.0, 0]}
    for g in groups or {}:
        sums[g] = [0.0, 0.0, 0]
    for starts in torch.arange(split.n_windows, device=dev).split(bs):
        x, m, y = split.batch(starts, var_idx)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype or torch.float32,
                            enabled=amp_dtype is not None):
            p = model(x, m, var_idx)
        err = p.float() - y
        for g, (se, ae, cnt) in list(sums.items()):
            e = err if g == "all" else err[..., groups[g]]
            sums[g] = [se + (e ** 2).sum().item(), ae + e.abs().sum().item(), cnt + e.numel()]
    out = {}
    for g, (se, ae, cnt) in sums.items():
        k = "" if g == "all" else f"_{g}"
        out[f"mse{k}"], out[f"mae{k}"] = se / cnt, ae / cnt
    return out


def run(cfg, log=print):
    """cfg: dict with data/model/training keys (see run.py). Returns a result dict."""
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    if cfg.get("tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.get("amp", "none")]
    (train, val, test), cols = load_splits(cfg["data"], cfg["root_path"], cfg["seq_len"],
                                           cfg["pred_len"], cfg["use_marks"], device)
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
        scan_order=cfg.get("scan_order", "fixed")).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # identity parameters start at zero and Adam moves them by about lr per step, so with the
    # small official learning rates (5e-5 .. 1e-3) they can be given a larger one
    id_params = list(model.identity.parameters())
    id_set = {id(p) for p in id_params}
    groups = [{"params": [p for p in model.parameters() if id(p) not in id_set], "mult": 1.0}]
    if id_params:
        groups.append({"params": id_params, "mult": cfg.get("id_lr_mult", 1.0)})
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
                loss = F.mse_loss(model(x, m, seen).float(), y)
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
    metrics = evaluate(model, test, None, amp_dtype=amp_dtype, groups=groups)
    k = cfg.get("eval_orders", 1)
    if k > 1:                                      # same model, scan-order averaging at test
        model.set_eval_orders(k)
        avg = evaluate(model, test, None, amp_dtype=amp_dtype, groups=groups)
        metrics.update({f"{name}_orders": v for name, v in avg.items()})
        model.set_eval_orders(1)
    result = dict(cfg, val_mse=best, best_epoch=best_epoch, epochs_run=epoch + 1,
                  train_time_s=round(train_time, 1), test_time_s=round(time.time() - t1, 1),
                  n_params=n_params, n_vars=n_vars, mamba_backend=model.mamba_backend(), **metrics)
    if device.type == "cuda":
        result["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    return result, model, cols
