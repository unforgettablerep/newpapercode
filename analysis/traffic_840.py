"""Experiment 5: share of the test MSE from sensor 840 and the 9 most harmed sensors.

python analysis/traffic_840.py <newpaper>/results/4090/traffic.jsonl
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
TOP = [341, 472, 567, 642, 668, 722, 724, 726, 840]
print("share of total test MSE (none, seed mean): sensor 840 / top9")
for model, b in (("itransformer", "none"), ("smamba", "none"), ("smamba", "none-epochs10")):
    t = [np.array(g[(model, b)][s]["per_var"]["test_mse"]) for s in range(3)]
    print(f"  {model} {b}: 840 {st.mean(x[840] / x.sum() * 100 for x in t):.1f}%  top9 {st.mean(x[TOP].sum() / x.sum() * 100 for x in t):.1f}%")
print("\nvariant: total dMSE% | without sensor 840 | without the 9 sensors | MAE% | MSE without 9 (abs)")
for model, b, vs in (("itransformer", "none", ["add", "add-drop0.1", "add-drop0.3", "lowrank-r8"]),
                     ("smamba", "none", ["add", "add-drop0.1", "add-drop0.3", "lowrank-r8"]),
                     ("smamba", "none-epochs10", ["add-epochs10"]), ("smamba", "none-shuf", ["add-shuf"])):
    for v in [b] + vs:
        res = collections.defaultdict(list)
        for s in range(3):
            A, B = g[(model, v)][s]["per_var"], g[(model, b)][s]["per_var"]
            ta, tb = np.array(A["test_mse"]), np.array(B["test_mse"])
            keep1 = np.ones(len(ta), bool); keep1[840] = False
            keep9 = np.ones(len(ta), bool); keep9[TOP] = False
            res["tot"].append((ta.mean() / tb.mean() - 1) * 100)
            res["w/o840"].append((ta[keep1].mean() / tb[keep1].mean() - 1) * 100)
            res["w/o9"].append((ta[keep9].mean() / tb[keep9].mean() - 1) * 100)
            res["mae"].append((np.mean(A["test_mae"]) / np.mean(B["test_mae"]) - 1) * 100)
            res["abs9"].append(ta[keep9].mean())
            res["better9"].append(ta[keep9].mean() < tb[keep9].mean())
        print(f"  {model} {v}: {st.mean(res['tot']):+.2f} | {st.mean(res['w/o840']):+.2f} | {st.mean(res['w/o9']):+.2f} ({sum(res['better9'])}/3) | {st.mean(res['mae']):+.2f} | {st.mean(res['abs9']):.4f}")
