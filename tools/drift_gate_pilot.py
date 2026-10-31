"""Pilot: label-free per-variable routing between an identity model and a no-identity model.

python tools/drift_gate_pilot.py <newpaper>/results/4090/traffic.jsonl

Uses runs made with --per-var. Variables whose input statistics drift from the training split
(|mean| and |log std| in training units) get the no-identity forecast, the others the identity
forecast. The routed fraction is picked on the validation split (validation drift, validation
per-variable MSE) and applied to the test split. Limitations: split-level statistics instead of
per-window ones, and two separately trained models instead of one gated model.
"""
import collections, json, statistics as st, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
def var(r):
    v = r["identity"] + (f"-drop{r['id_dropout']}" if r["id_dropout"] else "")
    return v + ("-" + r["tag"] if r.get("tag") else "") + ("-shuf" if r.get("scan_order") == "shuffle" else "")
g = collections.defaultdict(dict)
for r in rows:
    if "per_var" in r: g[(r["model"], var(r))][r["seed"]] = r["per_var"]
score = {
    "level": lambda p, s: np.abs(np.array(p[f"{s}_mean"])),
    "scale": lambda p, s: np.abs(np.log(np.array(p[f"{s}_std"]))),
    "both": lambda p, s: np.abs(np.array(p[f"{s}_mean"])) + np.abs(np.log(np.array(p[f"{s}_std"]))),
}
for model, a, b in (("itransformer", "add", "none"), ("smamba", "add", "none"), ("itransformer", "add-drop0.3", "none"), ("smamba", "add-drop0.1", "none")):
    res = collections.defaultdict(list)
    for s in range(3):
        A, B = g[(model, a)][s], g[(model, b)][s]
        ta, tb, va, vb = (np.array(x) for x in (A["test_mse"], B["test_mse"], A["val_mse"], B["val_mse"]))
        res["add"].append((ta.mean() / tb.mean() - 1) * 100)
        res["oracle"].append((np.minimum(ta, tb).mean() / tb.mean() - 1) * 100)
        for name, f in score.items():
            sv, stt = f(B, "val"), f(B, "test")
            best = None
            for q in np.linspace(0.5, 1.0, 51):          # gate the top (1-q) fraction by drift score
                tau = np.quantile(sv, q)
                comb_v = np.where(sv > tau, vb, va).mean()
                if best is None or comb_v < best[0]: best = (comb_v, q)
            q = best[1]; tau = np.quantile(stt, q)          # same fraction on the test split
            comb = np.where(stt > tau, tb, ta).mean()
            res[f"gate-{name}"].append((comb / tb.mean() - 1) * 100)
            res[f"q-{name}"].append(q)
        # fixed fractions on test-split drift score for reference
        for frac in (0.01, 0.02, 0.05):
            sc = score["both"](B, "test"); tau = np.quantile(sc, 1 - frac)
            res[f"fixed{frac:.0%}"].append((np.where(sc > tau, tb, ta).mean() / tb.mean() - 1) * 100)
    print(f"{model} {a} vs {b}: " + "; ".join(f"{k} {st.mean(v):+.2f}" + ("%" if not k.startswith("q-") else "") for k, v in res.items()))
