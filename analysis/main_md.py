"""Experiment 4: the markdown main table.

python analysis/main_md.py <newpaper>/results/4090/main.jsonl
"""
import collections, json, statistics as st, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
g = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in rows:
    g[(r["data"], r["pred_len"], r["model"])][r["identity"]][r["seed"]] = r
V = {"effective": "有效", "marginal": "微弱", "worse": "变差", "unclear": "不确定"}
def verdict(d):
    b = sum(x < 0 for x in d)
    if b == len(d): return "有效" if st.mean(d) <= -1 else "微弱"
    return "变差" if b == 0 else "不确定"
m = lambda d, k: st.mean(x[k] for x in d.values())
pd = lambda a, b, k: [(a[s][k] / b[s][k] - 1) * 100 for s in sorted(b)]
ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "Electricity", "Traffic"]
print("| 数据集 | H | S-Mamba 无 → 有 | ΔMSE | ΔMAE | 判定 | iTransformer 无 → 有 | ΔMSE | ΔMAE | 判定 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for data in ORDER:
    for H in (96, 192, 336, 720):
        cells = []
        for model in ("smamba", "itransformer"):
            n, a = g[(data, H, model)]["none"], g[(data, H, model)]["add"]
            d, dm = pd(a, n, "mse"), pd(a, n, "mae")
            v = verdict(d)
            b = "**" if v == "有效" else ""
            cells += [f"{m(n,'mse'):.4f} → {b}{m(a,'mse'):.4f}{b}", f"{st.mean(d):+.2f}%", f"{st.mean(dm):+.2f}%", v]
        print(f"| {data} | {H} | " + " | ".join(cells) + " |")
