"""Experiment 5: which Traffic sensors carry the identity loss; per-variable selection on validation.

python analysis/traffic_deep.py <newpaper>/results/4090/traffic.jsonl
"""
import collections, json, statistics as st, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
def var(r):
    v = r["identity"] + (f"-r{r['id_rank']}" if r["identity"] == "lowrank" else "")
    v += f"-drop{r['id_dropout']}" if r["id_dropout"] else ""
    return v + ("-" + r["tag"] if r.get("tag") else "") + ("-shuf" if r.get("scan_order") == "shuffle" else "")
g = collections.defaultdict(dict)
for r in rows:
    g[(r["model"], var(r))][r["seed"]] = r
pairs = [("itransformer", "add", "none"), ("itransformer", "add-drop0.1", "none"), ("itransformer", "add-drop0.3", "none"),
         ("smamba", "add", "none"), ("smamba", "add-drop0.1", "none"), ("smamba", "add-drop0.3", "none"),
         ("smamba", "add-epochs10", "none-epochs10"), ("smamba", "add-shuf", "none-shuf")]
for model, v, b in pairs:
    out = collections.defaultdict(list)
    for s in range(3):
        A, B = g[(model, v)][s]["per_var"], g[(model, b)][s]["per_var"]
        ta, tb = np.array(A["test_mse"]), np.array(B["test_mse"])
        va, vb = np.array(A["val_mse"]), np.array(B["val_mse"])
        d = ta - tb
        tot = d.sum()
        top = np.argsort(-d)[:9]
        out["total%"].append(tot / tb.sum() * 100)
        out["top9 share of increase"].append(d[top].sum() / tb.sum() * 100)
        out["test mae%"].append((np.mean(A["test_mae"]) / np.mean(B["test_mae"]) - 1) * 100)
        # per-variable choice by validation: use identity model where its val mse is lower
        pick = va < vb
        comb = np.where(pick, ta, tb)
        out["val-picked combo%"].append((comb.sum() / tb.sum() - 1) * 100)
        out["oracle combo%"].append((np.minimum(ta, tb).sum() / tb.sum() - 1) * 100)
        out["frac picked"].append(pick.mean() * 100)
        # are top-9 harmed on test also worse on val?
        out["top9 also worse on val"].append(float((va[top] > vb[top]).mean() * 100))
        out["top9 val rel change%"].append(float(np.median(va[top] / vb[top] - 1) * 100))
        out["top9 base test mse"].append(float(np.median(tb[top])))
        out["all base test mse median"].append(float(np.median(tb)))
        out["top9 ids"].append(tuple(sorted(top.tolist())))
    print(f"\n{model} {v} vs {b}: " + "; ".join(f"{k} {st.mean(x):+.2f}" for k, x in out.items() if k != "top9 ids"))
    c = collections.Counter(i for t in out["top9 ids"] for i in t)
    print("  top9 by absolute increase, count over seeds:", c.most_common(12))
# characterise the consistently harmed sensors
B = g[("itransformer", "none")][0]["per_var"]
ids = [851, 800, 46, 282, 311, 472, 341, 518, 224, 840, 668, 567]
ts, tm, vs, vm = (np.array(B[k]) for k in ("test_std", "test_mean", "val_std", "val_mean"))
print("\nall sensors: test_std quantiles", np.round(np.quantile(ts, [0.05, 0.25, 0.5, 0.75, 0.95]), 2), "val_std", np.round(np.quantile(vs, [0.05, 0.5, 0.95]), 2))
for i in ids:
    print(f"  sensor {i}: val mean {vm[i]:+.2f} std {vs[i]:.2f} | test mean {tm[i]:+.2f} std {ts[i]:.2f} | none test mse {B['test_mse'][i]:.3f} val mse {B['val_mse'][i]:.3f}")
