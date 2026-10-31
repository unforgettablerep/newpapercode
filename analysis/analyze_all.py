"""Experiments 2-4 (README): main table vs published numbers, ablation in both runs, scan-order hypothesis, run-to-run reproducibility.

python analysis/analyze_all.py <newpaper>/results/4090 [main|ablation|order|repro|all]
"""
import collections
import json
import statistics as st
import sys
from pathlib import Path

R = Path(sys.argv[1])
load = lambda f: [json.loads(l) for l in open(R / f) if l.strip()]

# published numbers (S-Mamba paper, Tables 2-4; the iTransformer column is copied from iTransformer)
PUB = {
    "smamba": {"Traffic": [(.382, .261), (.396, .267), (.417, .276), (.460, .300)],
               "ETTm1": [(.333, .368), (.376, .390), (.408, .413), (.475, .448)],
               "ETTm2": [(.179, .263), (.250, .309), (.312, .349), (.411, .406)],
               "ETTh1": [(.386, .405), (.443, .437), (.489, .468), (.502, .489)],
               "ETTh2": [(.296, .348), (.376, .396), (.424, .431), (.426, .444)],
               "Electricity": [(.139, .235), (.159, .255), (.176, .272), (.204, .298)],
               "Exchange": [(.086, .207), (.182, .304), (.332, .418), (.867, .703)],
               "Weather": [(.165, .210), (.214, .252), (.274, .297), (.350, .345)]},
    "itransformer": {"Traffic": [(.395, .268), (.417, .276), (.433, .283), (.467, .302)],
                     "ETTm1": [(.334, .368), (.377, .391), (.426, .420), (.491, .459)],
                     "ETTm2": [(.180, .264), (.250, .309), (.311, .348), (.412, .407)],
                     "ETTh1": [(.386, .405), (.441, .436), (.487, .458), (.503, .491)],
                     "ETTh2": [(.297, .349), (.380, .400), (.428, .432), (.427, .445)],
                     "Electricity": [(.148, .240), (.162, .253), (.178, .269), (.225, .317)],
                     "Exchange": [(.086, .206), (.177, .299), (.331, .417), (.847, .691)],
                     "Weather": [(.174, .214), (.221, .254), (.278, .296), (.358, .347)]},
}
HS = [96, 192, 336, 720]
ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "Electricity", "Traffic"]


def variant(r):
    v = r["identity"] + (f"-r{r['id_rank']}" if r["identity"] == "lowrank" else "")
    v += "" if r["id_loc"] == "input" else "-every"
    v += f"-drop{r['id_dropout']}" if r["id_dropout"] else ""
    v += f"-lr{r.get('id_lr_mult', 1):g}x" if r.get("id_lr_mult", 1) != 1 else ""
    return v


def verdict(d):
    b = sum(x < 0 for x in d)
    if b == len(d):
        return "有效" if st.mean(d) <= -1 else "微弱"
    return "变差" if b == 0 else "不确定"


def paired(a, b, m="mse"):
    s = sorted(set(a) & set(b))
    return [(a[i][m] / b[i][m] - 1) * 100 for i in s]


def mean(d, m="mse"):
    return st.mean(x[m] for x in d.values())


section = sys.argv[2] if len(sys.argv) > 2 else "all"

# ---------------------------------------------------------------- main table
if section in ("all", "main"):
    main = load("main.jsonl")
    g = collections.defaultdict(dict)
    for r in main:
        g[(r["data"], r["pred_len"], r["model"])].setdefault(variant(r), {})[r["seed"]] = r
    print("## main: data | H | model | published | none (mse/mae) | add | dMSE% | dMAE% | better | verdict | none-vs-pub%")
    agg = collections.defaultdict(list)
    for data in ORDER:
        for i, H in enumerate(HS):
            for model in ("smamba", "itransformer"):
                n, a = g[(data, H, model)]["none"], g[(data, H, model)]["add"]
                d, dm = paired(a, n), paired(a, n, "mae")
                pm, pa = PUB[model][data][i]
                gap = (mean(n) / pm - 1) * 100
                agg[(data, model)].append((st.mean(d), st.mean(dm), verdict(d), gap))
                print(f"{data}|{H}|{model}|{pm:.3f}/{pa:.3f}|{mean(n):.4f}/{mean(n,'mae'):.4f}|"
                      f"{mean(a):.4f}/{mean(a,'mae'):.4f}|{st.mean(d):+.2f}|{st.mean(dm):+.2f}|"
                      f"{sum(x<0 for x in d)}/{len(d)}|{verdict(d)}|{gap:+.1f}")
    print("\n## main aggregated over horizons: data | model | mean dMSE% | mean dMAE% | verdicts | mean none-vs-pub%")
    for data in ORDER:
        for model in ("smamba", "itransformer"):
            v = agg[(data, model)]
            c = collections.Counter(x[2] for x in v)
            print(f"{data}|{model}|{st.mean(x[0] for x in v):+.2f}|{st.mean(x[1] for x in v):+.2f}|"
                  f"{dict(c)}|{st.mean(x[3] for x in v):+.1f}")
    for model in ("smamba", "itransformer"):
        allv = [x for (d_, m_), v in agg.items() if m_ == model for x in v]
        c = collections.Counter(x[2] for x in allv)
        print(f"ALL|{model}|{st.mean(x[0] for x in allv):+.2f}|{st.mean(x[1] for x in allv):+.2f}|{dict(c)}")
    # best model per (data, H) among the four
    print("\n## best of 4 per data/H (mean mse)")
    wins = collections.Counter()
    for data in ORDER:
        for H in HS:
            cands = {(m, v): mean(g[(data, H, m)][v]) for m in ("smamba", "itransformer") for v in ("none", "add")}
            best = min(cands, key=cands.get)
            wins[best] += 1
            print(data, H, best, f"{cands[best]:.4f}", " | ".join(f"{m[0]}:{v}={x:.4f}" for (m, v), x in cands.items()))
    print("wins:", dict(wins))

# ---------------------------------------------------------------- ablation (two replicates)
if section in ("all", "ablation"):
    nb = load("newablation.jsonl")
    reps = {"rep1 (old code)": [r for r in nb if "mamba_backend" not in r],
            "rep2 (new code, cuda)": [r for r in nb if "mamba_backend" in r]}
    for name, rows in reps.items():
        g = collections.defaultdict(dict)
        for r in rows:
            g[(r["data"], r["train_var_ratio"])].setdefault(variant(r), {})[r["seed"]] = r
        print(f"\n## ablation {name}: data | ratio | variant | mse | dMSE% vs none | better | verdict | seen | unseen")
        for data in ("ETTh1", "Weather", "Electricity"):
            for ratio in (1.0, 0.4):
                base = g[(data, ratio)]["none"]
                for v, runs in sorted(g[(data, ratio)].items(), key=lambda kv: (kv[0] != "none", kv[0])):
                    d = paired(runs, base) if v != "none" else []
                    extra = ""
                    if ratio < 1:
                        ds = paired(runs, base, "mse_seen") if v != "none" else []
                        du = paired(runs, base, "mse_unseen") if v != "none" else []
                        extra = (f"|seen {mean(runs,'mse_seen'):.4f} ({st.mean(ds):+.2f}%)" if ds else f"|seen {mean(runs,'mse_seen'):.4f}") + \
                                (f"|unseen {mean(runs,'mse_unseen'):.4f} ({st.mean(du):+.2f}%)" if du else f"|unseen {mean(runs,'mse_unseen'):.4f}")
                    print(f"{data}|{ratio}|{v}|{mean(runs):.4f}|" + (f"{st.mean(d):+.2f}|{sum(x<0 for x in d)}/{len(d)}|{verdict(d)}" if d else "-|-|-") + extra)

# ---------------------------------------------------------------- order hypothesis
if section in ("all", "order"):
    order = load("order.jsonl")
    fixed_sources = {"quick": load("quick_T96.jsonl"),
                     "ablation rep1": [r for r in load("newablation.jsonl") if "mamba_backend" not in r],
                     "ablation rep2": [r for r in load("newablation.jsonl") if "mamba_backend" in r]}
    sh = collections.defaultdict(dict)
    for r in order:
        sh[r["data"]].setdefault(r["identity"], {})[r["seed"]] = r
    print("\n## order: shuffled-order S-Mamba (orig-order test / 4-order avg)")
    for data in ("ETTh1", "Weather", "Electricity"):
        n, a = sh[data]["none"], sh[data]["add"]
        print(f"{data}: none {mean(n):.4f} / {mean(n,'mse_orders'):.4f}; add {mean(a):.4f} / {mean(a,'mse_orders'):.4f}; "
              f"id gain {st.mean(paired(a,n)):+.2f}% ({sum(x<0 for x in paired(a,n))}/3) / "
              f"{st.mean(paired(a,n,'mse_orders')):+.2f}% ({sum(x<0 for x in paired(a,n,'mse_orders'))}/3); "
              f"avg-vs-orig none {st.mean([(n[s]['mse_orders']/n[s]['mse']-1)*100 for s in n]):+.2f}% add {st.mean([(a[s]['mse_orders']/a[s]['mse']-1)*100 for s in a]):+.2f}%")
        for src, rows in fixed_sources.items():
            f = collections.defaultdict(dict)
            for r in rows:
                if r["model"] == "smamba" and r["data"] == data and r["pred_len"] == 96 and r.get("train_var_ratio", 1) == 1 \
                        and variant(r) in ("none", "add"):
                    f[variant(r)][r["seed"]] = r
            fn, fa = f["none"], f["add"]
            print(f"   fixed[{src}]: none {mean(fn):.4f} add {mean(fa):.4f} id gain {st.mean(paired(fa,fn)):+.2f}% "
                  f"({sum(x<0 for x in paired(fa,fn))}/3); shuffle cost w/o id {st.mean(paired(n,fn)):+.2f}% "
                  f"({sum(x>0 for x in paired(n,fn))}/3 worse), shuffle cost with id {st.mean(paired(a,fa)):+.2f}%; "
                  f"shuffled+id vs fixed none {st.mean(paired(a,fn)):+.2f}%")

# ---------------------------------------------------------------- reproducibility
if section in ("all", "repro"):
    def key(r):
        return (r["data"], r["pred_len"], r["model"], variant(r), r.get("train_var_ratio", 1), r["seed"])
    srcs = {"quick": load("quick_T96.jsonl"), "main": load("main.jsonl"),
            "abl1": [r for r in load("newablation.jsonl") if "mamba_backend" not in r],
            "abl2": [r for r in load("newablation.jsonl") if "mamba_backend" in r]}
    idx = {k: {key(r): r for r in v if r.get("scan_order", "fixed") == "fixed"} for k, v in srcs.items()}
    print("\n## reproducibility: same config and seed in two files")
    for a_, b_ in [("quick", "main"), ("quick", "abl1"), ("quick", "abl2"), ("abl1", "abl2"), ("main", "abl1")]:
        common = sorted(set(idx[a_]) & set(idx[b_]))
        if not common:
            continue
        d = [abs(idx[b_][k]["mse"] / idx[a_][k]["mse"] - 1) * 100 for k in common]
        ex = sum(idx[b_][k]["mse"] == idx[a_][k]["mse"] for k in common)
        worst = max(common, key=lambda k: abs(idx[b_][k]["mse"] / idx[a_][k]["mse"] - 1))
        print(f"{a_} vs {b_}: {len(common)} pairs, identical {ex}, mean |d| {st.mean(d):.3f}%, median {st.median(d):.3f}%, "
              f"max {max(d):.2f}% at {worst}")
