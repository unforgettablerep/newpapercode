"""Experiment 2: shuffled-order S-Mamba against fixed order and iTransformer (H=96).

python analysis/order_extra.py <newpaper>/results/4090
"""
import collections, json, statistics as st, sys
from pathlib import Path
R = Path(sys.argv[1])
load = lambda f: [json.loads(l) for l in open(R / f) if l.strip()]
order = load("order.jsonl"); quick = load("quick_T96.jsonl")
nb = load("newablation.jsonl"); rep2 = [r for r in nb if "mamba_backend" in r]
def grp(rows, model, scan):
    g = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        if r["model"] == model and r["pred_len"] == 96 and r.get("scan_order", "fixed") == scan \
           and r.get("train_var_ratio", 1) == 1 and r["id_loc"] == "input" and not r["id_dropout"] \
           and r.get("id_lr_mult", 1) == 1 and r["identity"] in ("none", "add"):
            g[r["data"]][r["identity"]][r["seed"]] = r
    return g
sh = grp(order, "smamba", "shuffle"); fq = grp(quick, "smamba", "fixed"); f2 = grp(rep2, "smamba", "fixed")
it = grp(quick, "itransformer", "fixed")
m = lambda d, k="mse": st.mean(x[k] for x in d.values())
pd = lambda a, b, ka="mse", kb="mse": [(a[s][ka] / b[s][kb] - 1) * 100 for s in sorted(set(a) & set(b))]
f = lambda d: f"{st.mean(d):+.2f}% ({sum(x<0 for x in d)}/{len(d)})"
for data in ("ETTh1", "Weather", "Electricity"):
    n, a = sh[data]["none"], sh[data]["add"]
    print(f"\n{data}: shuffled none {m(n):.4f}/{m(n,'mae'):.4f} avg4 {m(n,'mse_orders'):.4f}/{m(n,'mae_orders'):.4f}; "
          f"add {m(a):.4f}/{m(a,'mae'):.4f} avg4 {m(a,'mse_orders'):.4f}/{m(a,'mae_orders'):.4f}")
    print("  epochs best/run none", [(r['best_epoch'], r['epochs_run']) for r in n.values()], "add", [(r['best_epoch'], r['epochs_run']) for r in a.values()])
    print("  train time none", st.mean(r['train_time_s'] for r in n.values()), "add", st.mean(r['train_time_s'] for r in a.values()),
          "| fixed quick none", st.mean(r['train_time_s'] for r in fq[data]['none'].values()), "add", st.mean(r['train_time_s'] for r in fq[data]['add'].values()))
    print("  val mse none", m(n, 'val_mse'), "add", m(a, 'val_mse'), "| fixed none", m(fq[data]['none'], 'val_mse'), "add", m(fq[data]['add'], 'val_mse'))
    print("  MAE id gain shuffled", f(pd(a, n, 'mae', 'mae')), " avg4", f(pd(a, n, 'mae_orders', 'mae_orders')))
    for src, F in (("quick", fq), ("rep2", f2)):
        fn, fa = F[data]["none"], F[data]["add"]
        print(f"  [{src}] fixed none {m(fn):.4f} add {m(fa):.4f} | shuf+id avg4 vs fixed none {f(pd(a, fn, 'mse_orders'))}, vs fixed add {f(pd(a, fa, 'mse_orders'))}"
              f" | shuf none avg4 vs fixed none {f(pd(n, fn, 'mse_orders'))}")
    print(f"  iTransformer quick none {m(it[data]['none']):.4f} add {m(it[data]['add']):.4f}; shuf+id (orig) vs iT+id {f(pd(a, it[data]['add']))}, avg4 vs iT+id {f(pd(a, it[data]['add'], 'mse_orders'))}")
