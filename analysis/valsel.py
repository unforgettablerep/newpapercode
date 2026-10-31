"""Experiments 5 and 7: choosing the identity variant on the validation split.

python analysis/valsel.py <newpaper>
"""
import collections, json, statistics as st, sys
from pathlib import Path
N = Path(sys.argv[1])
load = lambda p: [json.loads(l) for l in open(N / p) if l.strip()]
def var(r):
    v = r["identity"] + (f"-r{r['id_rank']}" if r["identity"] == "lowrank" else "")
    v += f"-drop{r['id_dropout']}" if r["id_dropout"] else ""
    v += f"-lr{r['id_lr_mult']:g}x" if r.get("id_lr_mult", 1) != 1 else ""
    return v + ("-" + r["tag"] if r.get("tag") else "") + ("-shuf" if r.get("scan_order", "fixed") == "shuffle" else "")
g = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in load("results/4090/main.jsonl")[:384] + load("results/4090/idlr.jsonl"):
    g[(r["data"], r["pred_len"], r["model"])][var(r)][r["seed"]] = r
m = lambda d, k: st.mean(x[k] for x in d.values())
def verdict(d):
    b = sum(x < 0 for x in d)
    if b == len(d): return "有效" if st.mean(d) <= -1 else "微弱"
    return "变差" if b == 0 else "不确定"
print("== choose among none / add / add-lr10x by validation ==")
tot = collections.defaultdict(list)
for data in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather"]:
    for model in ("smamba", "itransformer"):
        line = []
        for H in (96, 192, 336, 720):
            c = g[(data, H, model)]
            pick = min(("none", "add", "add-lr10x"), key=lambda v: m(c[v], "val_mse"))
            d = [(c[pick][s]["mse"] / c["none"][s]["mse"] - 1) * 100 for s in c["none"]]
            best = min(("none", "add", "add-lr10x"), key=lambda v: m(c[v], "mse"))
            tot[model].append(st.mean(d))
            line.append(f"H{H}:{pick}({st.mean(d):+.2f}%,{verdict(d) if pick != 'none' else '-'};test-best {best})")
        print(data, model, " ".join(line))
for model, v in tot.items():
    print(f"mean over 24 ({model}): {st.mean(v):+.2f}%; picks with test worse than none: {sum(x > 0 for x in v)}")
print("\n== Traffic H=96 (traffic.jsonl): choose among none / add / drop0.1 / drop0.3 / lowrank by validation ==")
t = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in load("results/4090/traffic.jsonl"):
    if not r.get("tag") and r.get("scan_order") == "fixed":
        t[r["model"]][var(r)][r["seed"]] = r
for model, c in t.items():
    vals = {v: m(c[v], "val_mse") for v in c}
    pick = min(vals, key=vals.get)
    d = [(c[pick][s]["mse"] / c["none"][s]["mse"] - 1) * 100 for s in c["none"]]
    print(model, "val:", {v: round(x, 4) for v, x in sorted(vals.items(), key=lambda kv: kv[1])}, "-> pick", pick, f"test {st.mean(d):+.2f}% ({sum(x<0 for x in d)}/3)")
