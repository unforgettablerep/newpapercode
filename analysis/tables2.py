"""Experiments 7-8: markdown tables (shuffle main table and aggregates, identity lr x10).

python analysis/tables2.py <newpaper> [shuffle|shuffle_agg|idlr]
"""
import collections, json, statistics as st, sys
from pathlib import Path
N = Path(sys.argv[1])
load = lambda p: [json.loads(l) for l in open(N / p) if l.strip()]
HS = [96, 192, 336, 720]
ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "Electricity", "Traffic"]
PUB = {"Traffic": [.382, .396, .417, .460], "ETTm1": [.333, .376, .408, .475], "ETTm2": [.179, .250, .312, .411],
       "ETTh1": [.386, .443, .489, .502], "ETTh2": [.296, .376, .424, .426], "Electricity": [.139, .159, .176, .204],
       "Exchange": [.086, .182, .332, .867], "Weather": [.165, .214, .274, .350]}
def verdict(d):
    b = sum(x < 0 for x in d)
    if b == len(d): return "有效" if st.mean(d) <= -1 else "微弱"
    return "变差" if b == 0 else "不确定"
def pd(a, b, ka="mse", kb="mse"):
    return [(a[s][ka] / b[s][kb] - 1) * 100 for s in sorted(set(a) & set(b))]
m = lambda d, k="mse": st.mean(x[k] for x in d.values())
def idx(rows, cond):
    g = collections.defaultdict(dict)
    for r in rows:
        if cond(r): g[(r["data"], r["pred_len"], r["model"])][r["seed"]] = r
    return g
main = load("results/4090/main.jsonl")[:384]
mn = idx(main, lambda r: r["identity"] == "none"); ma = idx(main, lambda r: r["identity"] == "add")
sm = load("results/4090/shuffle_main.jsonl")
sn = idx(sm, lambda r: r["identity"] == "none"); sa = idx(sm, lambda r: r["identity"] == "add")
fm = lambda x: f"{x:+.2f}%".replace("-", "−")
if sys.argv[2] == "shuffle":
    print("| 数据集 | H | 论文 S-Mamba | 固定顺序 无身份 | 固定顺序 有身份 | 随机顺序+身份 原顺序 | K=2 | K=8 | K=8 对固定无身份 | 判定 | MAE 变化 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for data in ORDER:
        for i, H in enumerate(HS):
            k = (data, H, "smamba"); a = sa[k]
            d = pd(a, mn[k], "mse_orders8"); dm = pd(a, mn[k], "mae_orders8", "mae"); v = verdict(d)
            b = "**" if v == "有效" else ""
            print(f"| {data} | {H} | {PUB[data][i]:.3f} | {m(mn[k]):.4f} | {m(ma[k]):.4f} | {m(a):.4f} | {m(a,'mse_orders2'):.4f} | "
                  f"{b}{m(a,'mse_orders8'):.4f}{b} | {fm(st.mean(d))} | {v} | {fm(st.mean(dm))} |")
if sys.argv[2] == "shuffle_agg":
    print("| 数据集 | K=8 对固定无身份 | 判定（4 个长度） | 对固定有身份 | 只用原顺序（单次前向） | 随机顺序下身份的收益 | 无身份时随机顺序的代价 |")
    print("|---|---|---|---|---|---|---|")
    allv = []
    for data in ORDER:
        rows = []
        for H in HS:
            k = (data, H, "smamba"); a, n = sa[k], sn[k]
            d = pd(a, mn[k], "mse_orders8")
            rows.append((st.mean(d), verdict(d), st.mean(pd(a, ma[k], "mse_orders8")), st.mean(pd(a, mn[k])),
                         st.mean(pd(a, n, "mse_orders8", "mse_orders8")), st.mean(pd(n, mn[k]))))
        allv += rows
        c = collections.Counter(x[1] for x in rows)
        print(f"| {data} | {fm(st.mean(x[0] for x in rows))} | " + "、".join(f"{k} {v}" for k, v in c.most_common()) + " | "
              + " | ".join(fm(st.mean(x[i] for x in rows)) for i in (2, 3, 4, 5)) + " |")
    for name, sub in (("全部 32 组", allv), ("不含 Traffic 的 28 组", allv[:28])):
        c = collections.Counter(x[1] for x in sub)
        print(f"| {name} | {fm(st.mean(x[0] for x in sub))} | " + "、".join(f"{k} {v}" for k, v in c.most_common()) + " | "
              + " | ".join(fm(st.mean(x[i] for x in sub)) for i in (2, 3, 4, 5)) + " |")
if sys.argv[2] == "idlr":
    il = idx(load("results/4090/idlr.jsonl"), lambda r: True)
    print("| 数据集 | H | S-Mamba：`add` | ×10 | 判定 | iTransformer：`add` | ×10 | 判定 |")
    print("|---|---|---|---|---|---|---|---|")
    for data in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather"]:
        for H in HS:
            cells = []
            for model in ("smamba", "itransformer"):
                k = (data, H, model); d1, d0 = pd(il[k], mn[k]), pd(ma[k], mn[k]); v = verdict(d1)
                b = "**" if v == "有效" else ""
                cells += [fm(st.mean(d0)), f"{b}{fm(st.mean(d1))}{b}", v]
            print(f"| {data} | {H} | " + " | ".join(cells) + " |")
