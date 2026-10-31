"""Experiments 6-8: column permutation / 10 epochs, identity lr x10, shuffled order on every dataset.

python analysis/round2.py <newpaper> [perm|idlr|shuffle]
"""
import collections, json, statistics as st, sys
from pathlib import Path
N = Path(sys.argv[1]); sec = sys.argv[2]
load = lambda p: [json.loads(l) for l in open(N / p) if l.strip()]
HS = [96, 192, 336, 720]
def verdict(d):
    b = sum(x < 0 for x in d)
    if b == len(d): return "有效" if st.mean(d) <= -1 else "微弱"
    return "变差" if b == 0 else "不确定"
def pd(a, b, ka="mse", kb=None):
    kb = kb or ka; s = sorted(set(a) & set(b))
    return [(a[i][ka] / b[i][kb] - 1) * 100 for i in s]
f = lambda d: f"{st.mean(d):+.2f}% ({sum(x<0 for x in d)}/{len(d)}) {verdict(d)}"
m = lambda d, k="mse": st.mean(x[k] for x in d.values())
def idx(rows, cond):
    g = collections.defaultdict(dict)
    for r in rows:
        if cond(r): g[(r["data"], r["pred_len"], r["model"])][r["seed"]] = r
    return g
main = load("results/4090/main.jsonl")[:384]
plain = lambda r: r["id_loc"] == "input" and not r["id_dropout"] and r.get("id_lr_mult", 1) == 1 and r.get("scan_order", "fixed") == "fixed"
mn = idx(main, lambda r: r["identity"] == "none" and plain(r))
ma = idx(main, lambda r: r["identity"] == "add" and plain(r))
ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "Electricity", "Traffic"]

if sec == "perm":
    q = load("results/4090/quick_T96.jsonl"); pr = load("results/4090/perm.jsonl"); od = load("results/4090/order.jsonl")
    qn = idx(q, lambda r: r["model"] == "smamba" and r["identity"] == "none"); qa = idx(q, lambda r: r["model"] == "smamba" and r["identity"] == "add")
    pn = idx(pr, lambda r: r.get("var_perm") == 1 and r["identity"] == "none"); pa = idx(pr, lambda r: r.get("var_perm") == 1 and r["identity"] == "add")
    en = idx(pr, lambda r: r.get("tag") == "epochs10" and r["identity"] == "none"); ea = idx(pr, lambda r: r.get("tag") == "epochs10" and r["identity"] == "add")
    on = idx(od, lambda r: r["identity"] == "none"); oa = idx(od, lambda r: r["identity"] == "add")
    for data in ("ETTh1", "Weather", "Electricity"):
        k = (data, 96, "smamba")
        print(f"{data}: file none {m(qn[k]):.4f} add {m(qa[k]):.4f} | perm none {m(pn[k]):.4f} add {m(pa[k]):.4f}")
        print(f"   perm vs file: none {f(pd(pn[k], qn[k]))}; add {f(pd(pa[k], qa[k]))}; id gain in perm {f(pd(pa[k], pn[k]))}")
        if k in en:
            print(f"   shuffle 10ep: none {m(en[k]):.4f}/{m(en[k],'mse_orders'):.4f} add {m(ea[k]):.4f}/{m(ea[k],'mse_orders'):.4f}; "
                  f"10ep vs 5ep add avg4 {f(pd(ea[k], oa[k], 'mse_orders'))}; 10ep add avg4 vs file add {f(pd(ea[k], qa[k], 'mse_orders', 'mse'))}; vs perm add {f(pd(ea[k], pa[k], 'mse_orders', 'mse'))}")
            print("   best epochs 10ep none", [(r['best_epoch'], r['epochs_run']) for r in en[k].values()], "add", [(r['best_epoch'], r['epochs_run']) for r in ea[k].values()])

if sec == "idlr":
    il = idx(load("results/4090/idlr.jsonl"), lambda r: True)
    print("data|H|model|none|add|add-lr10x|lr10x vs none|add vs none|lr10x vs add")
    agg = collections.defaultdict(list)
    for data in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather"]:
        for H in HS:
            for model in ("smamba", "itransformer"):
                k = (data, H, model)
                d1, d0, d2 = pd(il[k], mn[k]), pd(ma[k], mn[k]), pd(il[k], ma[k])
                agg[(data, model)].append((st.mean(d1), st.mean(d0), verdict(d1)))
                print(f"{data}|{H}|{model}|{m(mn[k]):.4f}|{m(ma[k]):.4f}|{m(il[k]):.4f}|{f(d1)}|{f(d0)}|{f(d2)}"
                      f"|best ep {[r['best_epoch'] for r in il[k].values()]} vs {[r['best_epoch'] for r in ma[k].values()]}")
    print("\nagg data|model|lr10x vs none mean|add vs none mean|verdicts")
    for (data, model), v in agg.items():
        print(data, model, f"{st.mean(x[0] for x in v):+.2f}", f"{st.mean(x[1] for x in v):+.2f}", dict(collections.Counter(x[2] for x in v)))

if sec == "shuffle":
    sm = load("results/4090/shuffle_main.jsonl")
    sn = idx(sm, lambda r: r["identity"] == "none"); sa = idx(sm, lambda r: r["identity"] == "add")
    print("data|H|fixed none|fixed add|shuf none orig/8|shuf add orig/2/4/8|shuf+id avg8 vs fixed none|vs fixed add|id gain shuffled(avg8)|shuffle cost w/o id (orig)")
    agg = collections.defaultdict(list)
    for data in ORDER:
        for H in HS:
            k = (data, H, "smamba")
            a, n = sa[k], sn[k]
            r1 = pd(a, mn[k], "mse_orders8", "mse"); r2 = pd(a, ma[k], "mse_orders8", "mse")
            r3 = pd(a, n, "mse_orders8"); r4 = pd(n, mn[k]); r5 = pd(a, mn[k], "mse", "mse")
            r6 = [(a[s]["mse_orders8"] / a[s]["mse"] - 1) * 100 for s in a]
            k4 = pd(a, mn[k], "mse_orders4", "mse"); k2 = pd(a, mn[k], "mse_orders2", "mse")
            agg[data].append((st.mean(r1), verdict(r1), st.mean(r2), st.mean(r3), st.mean(r4), st.mean(r5), st.mean(k2), st.mean(k4), st.mean(r6)))
            print(f"{data}|{H}|{m(mn[k]):.4f}|{m(ma[k]):.4f}|{m(n):.4f}/{m(n,'mse_orders8'):.4f}|{m(a):.4f}/{m(a,'mse_orders2'):.4f}/{m(a,'mse_orders4'):.4f}/{m(a,'mse_orders8'):.4f}"
                  f"|{f(r1)}|{f(r2)}|{f(r3)}|{f(r4)}| best ep {[r['best_epoch'] for r in a.values()]}/{a[0]['epochs']}")
    print("\nagg data | shuf+id avg8 vs fixed none | verdicts | vs fixed add | id gain shuffled | shuffle cost | shuf+id orig vs fixed none | K=2 | K=4 | avg8 vs orig")
    for data, v in agg.items():
        print(data, " | ".join(f"{st.mean(x[i] for x in v):+.2f}" if i != 1 else str(dict(collections.Counter(x[1] for x in v))) for i in range(9)))
    allv = [x for v in agg.values() for x in v]
    print("ALL", " | ".join(f"{st.mean(x[i] for x in allv):+.2f}" if i != 1 else str(dict(collections.Counter(x[1] for x in allv))) for i in range(9)))
    allv = [x for d, v in agg.items() if d != "Traffic" for x in v]
    print("noTraffic", " | ".join(f"{st.mean(x[i] for x in allv):+.2f}" if i != 1 else str(dict(collections.Counter(x[1] for x in allv))) for i in range(9)))
