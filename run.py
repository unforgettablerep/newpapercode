"""Train and test one configuration; append the result as one JSON line.

Examples
  python run.py --model smamba --data Electricity --pred-len 96 --identity add --seed 0
  python run.py --model itransformer --data Weather --pred-len 336 --identity lowrank --id-rank 8
  python run.py --model smamba --data Electricity --train-var-ratio 0.4 --identity add --id-dropout 0.3
  python run.py --model smamba --data Electricity --scan-order shuffle --identity add --eval-orders 2,4,8

Defaults for every dataset come from idembed/configs.py; any flag below overrides them.
"""
import argparse
import json
import os
from pathlib import Path
import torch
from idembed.configs import get_config
from idembed.engine import run


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["smamba", "itransformer"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--root-path", default="./data")
    ap.add_argument("--pred-len", type=int, default=96)
    ap.add_argument("--seq-len", type=int)
    ap.add_argument("--seed", type=int, default=0)
    # identity embedding
    ap.add_argument("--identity", choices=["none", "add", "lowrank"], default="none")
    ap.add_argument("--id-rank", type=int, default=8)
    ap.add_argument("--id-loc", choices=["input", "every"], default="input")
    ap.add_argument("--id-dropout", type=float, default=0.0)
    ap.add_argument("--id-lr-mult", type=float, default=1.0,
                    help="learning-rate multiplier for the identity parameters")
    ap.add_argument("--train-var-ratio", type=float, default=1.0,
                    help="train/validate on this fraction of the variables, test on all")
    # S-Mamba scan order
    ap.add_argument("--scan-order", choices=["fixed", "shuffle"], default="fixed",
                    help="shuffle: random token scan order at every training step (S-Mamba)")
    ap.add_argument("--eval-orders", default="1",
                    help="also test with the mixing averaged over this many random scan orders; "
                         "a comma-separated list (e.g. 2,4,8) tests each")
    ap.add_argument("--var-perm", type=int, default=None,
                    help="list the variables in a fixed random order (this seed) instead of the "
                         "file's column order")
    ap.add_argument("--per-var", action="store_true",
                    help="store per-variable test/validation errors and split statistics")
    # model / training overrides
    for name, typ in [("d-model", int), ("d-ff", int), ("e-layers", int), ("n-heads", int),
                      ("d-state", int), ("dropout", float), ("lr", float), ("batch-size", int),
                      ("epochs", int), ("patience", int)]:
        ap.add_argument(f"--{name}", type=typ)
    ap.add_argument("--no-marks", action="store_true", help="do not add calendar-feature tokens")
    ap.add_argument("--no-norm", action="store_true", help="disable instance normalisation")
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 matmuls (faster on RTX 4090)")
    ap.add_argument("--mamba-impl", choices=["auto", "cuda", "torch"], default="auto")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/results.jsonl")
    ap.add_argument("--save-ckpt", default=None, help="directory for the best checkpoint")
    ap.add_argument("--tag", default="", help="free-text label stored with the result")
    return ap.parse_args()


def main():
    a = parse()
    ks = [int(k) for k in a.eval_orders.split(",")]
    eval_orders = ks[0] if len(ks) == 1 else ks
    cfg = get_config(a.model, a.data)
    cfg.update(model=a.model, data=a.data, root_path=a.root_path, pred_len=a.pred_len, seed=a.seed,
               identity=a.identity, id_rank=a.id_rank, id_loc=a.id_loc, id_dropout=a.id_dropout,
               id_lr_mult=a.id_lr_mult, scan_order=a.scan_order, eval_orders=eval_orders,
               train_var_ratio=a.train_var_ratio, use_marks=not a.no_marks,
               amp=a.amp, tf32=a.tf32, mamba_impl=a.mamba_impl, device=a.device, tag=a.tag)
    if a.no_norm:
        cfg["use_norm"] = False
    if a.var_perm is not None:
        cfg["var_perm"] = a.var_perm
    if a.per_var:
        cfg["per_var"] = True
    for k in ["seq_len", "d_model", "d_ff", "e_layers", "n_heads", "d_state", "dropout", "lr",
              "batch_size", "epochs", "patience"]:
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v
    print(json.dumps({k: cfg[k] for k in sorted(cfg)}))
    result, model, cols = run(cfg)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"test mse {result['mse']:.4f}  mae {result['mae']:.4f}  (best epoch {result['best_epoch']}, "
          f"mamba backend {result['mamba_backend']})")
    for k in sorted(k for k in ks if k > 1):
        sfx = f"_orders{k}" if f"mse_orders{k}" in result else "_orders"
        print(f"averaged over {k} scan orders: mse {result['mse' + sfx]:.4f}  "
              f"mae {result['mae' + sfx]:.4f}")
    if a.save_ckpt:
        os.makedirs(a.save_ckpt, exist_ok=True)
        name = (f"{a.model}_{a.data}_{a.pred_len}_{a.identity}_{a.id_loc}_r{a.id_rank}"
                f"_p{a.id_dropout}_m{a.id_lr_mult}_v{a.train_var_ratio}_{a.scan_order}"
                + ("" if a.var_perm is None else f"_perm{a.var_perm}") + f"_s{a.seed}.pt")
        torch.save({"config": cfg, "state_dict": model.state_dict(), "columns": cols},
                   Path(a.save_ckpt) / name)


if __name__ == "__main__":
    main()
