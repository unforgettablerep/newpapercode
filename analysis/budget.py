"""papers/theory.md section 5: learning budget of the identity parameters vs their gains.

python analysis/budget.py <newpaper>/results/4090
"""
import collections, json, statistics as st, sys
R = sys.argv[1]
TRAIN_ROWS = {"ETTh1": 8640, "ETTh2": 8640, "ETTm1": 34560, "ETTm2": 34560, "Weather": 36887,
              "Electricity": 18412, "Traffic": 12280, "Exchange": 5311}
def budget(r):
    S = (TRAIN_ROWS[r["data"]] - r["seq_len"] - r["pred_len"] + 1) // r["batch_size"]
    E = r["best_epoch"]
    return r["lr"] * r.get("id_lr_mult", 1.0) * S * (3 - 2 ** (2 - E))
rows = [json.loads(l) for f in ("main.jsonl", "idlr.jsonl") for l in open(f"{R}/{f}") if l.strip()]
g = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in rows[:384 + 144]:
    v = r["identity"] + ("-x10" if r.get("id_lr_mult", 1) != 1 else "")
    g[(r["data"], r["pred_len"], r["model"])][v][r["seed"]] = r
out = []
for k, d in sorted(g.items()):
    if "add-x10" not in d: continue
    n, a, a10 = d["none"], d["add"], d["add-x10"]
    da = st.mean((a[s]["mse"] / n[s]["mse"] - 1) * 100 for s in n)
    d10 = st.mean((a10[s]["mse"] / a[s]["mse"] - 1) * 100 for s in n)
    out.append((k, st.mean(budget(r) for r in a.values()), st.mean(budget(r) for r in a10.values()), da, d10))
print("data|H|model|B(x1)|B(x10)|add vs none %|x10 vs x1 %")
for k, b1, b10, da, d10 in out:
    print(f"{k[0]}|{k[1]}|{k[2]}|{b1:.3f}|{b10:.2f}|{da:+.2f}|{d10:+.2f}")
print("\nby dataset: mean B(x1), mean add gain, mean x10-vs-x1")
for data in TRAIN_ROWS:
    xs = [o for o in out if o[0][0] == data]
    if xs: print(f"  {data}: B={st.mean(o[1] for o in xs):.3f}  add {st.mean(o[3] for o in xs):+.2f}%  x10 {st.mean(o[4] for o in xs):+.2f}%")
# Electricity / Traffic (no x10 runs): budget at x1
for k, d in sorted(g.items()):
    if k[0] in ("Electricity", "Traffic") and k[1] == 96 and "add" in d:
        print("  ", k, "B(x1) =", round(st.mean(budget(r) for r in d["add"].values()), 2))
