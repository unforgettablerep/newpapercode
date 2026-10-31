"""Experiments 9-10: the official-settings view over all result files (baselines vs published, identity / dropout / shuffle variants, recipes, validation selection, comparison table).

python analysis/off.py <newpaper> <newpapercode> [base|identity|select|rules|final|final_md|exp9|exp10] [out.json]
"""
import collections, json, statistics as st, sys
from pathlib import Path
N, C, sec = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(C)); sys.path.insert(0, str(C / "tools"))
from idembed.configs import get_config, changed
from summarize import training, TRAINING_KEYS
R = N / "results/4090"
files = ["main.jsonl", "official.jsonl", "idlr.jsonl", "shuffle_main.jsonl", "dropout.jsonl", "traffic.jsonl"]
if (N / "official.jsonl").exists():
    files = [f if (R / f).exists() else "../../" + f for f in files]
rows = []
for f in files:
    p = R / f
    if p.exists():
        rows += [dict(json.loads(l), _file=f.split("/")[-1]) for l in open(p) if l.strip()]
HS = [96, 192, 336, 720]
ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "Electricity", "Traffic"]
PUB = {"smamba": {"Traffic": [(.382, .261), (.396, .267), (.417, .276), (.460, .300)], "ETTm1": [(.333, .368), (.376, .390), (.408, .413), (.475, .448)],
                  "ETTm2": [(.179, .263), (.250, .309), (.312, .349), (.411, .406)], "ETTh1": [(.386, .405), (.443, .437), (.489, .468), (.502, .489)],
                  "ETTh2": [(.296, .348), (.376, .396), (.424, .431), (.426, .444)], "Electricity": [(.139, .235), (.159, .255), (.176, .272), (.204, .298)],
                  "Exchange": [(.086, .207), (.182, .304), (.332, .418), (.867, .703)], "Weather": [(.165, .210), (.214, .252), (.274, .297), (.350, .345)]},
       "itransformer": {"Traffic": [(.395, .268), (.417, .276), (.433, .283), (.467, .302)], "ETTm1": [(.334, .368), (.377, .391), (.426, .420), (.491, .459)],
                        "ETTm2": [(.180, .264), (.250, .309), (.311, .348), (.412, .407)], "ETTh1": [(.386, .405), (.441, .436), (.487, .458), (.503, .491)],
                        "ETTh2": [(.297, .349), (.380, .400), (.428, .432), (.427, .445)], "Electricity": [(.148, .240), (.162, .253), (.178, .269), (.225, .317)],
                        "Exchange": [(.086, .206), (.177, .299), (.331, .417), (.847, .691)], "Weather": [(.174, .214), (.221, .254), (.278, .296), (.358, .347)]}}
def off_hp(model, data, H):
    c = get_config(model, data, H); return tuple(c.get(k) for k in TRAINING_KEYS)
def var(r):
    v = r["identity"] + (f"-r{r['id_rank']}" if r["identity"] == "lowrank" else "")
    v += "" if r["id_loc"] == "input" else "-every"
    v += f"-drop{r['id_dropout']}" if r["id_dropout"] else ""
    v += f"-lr{r['id_lr_mult']:g}x" if r.get("id_lr_mult", 1) != 1 else ""
    return v + ("-shuf" if r.get("scan_order", "fixed") == "shuffle" else "")
# official view: (data, H, model) -> variant -> seed -> row ; first file wins
G = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in rows:
    if r.get("tag") or r.get("var_perm") is not None or r.get("train_var_ratio", 1) != 1:
        continue
    k = (r["data"], r["pred_len"], r["model"])
    if training(r) != off_hp(*[k[2], k[0], k[1]]):
        continue
    G[k][var(r)].setdefault(r["seed"], r)
def verdict(d):
    b = sum(x < 0 for x in d)
    if b == len(d): return "有效" if st.mean(d) <= -1 else "微弱"
    return "变差" if b == 0 else "不确定"
def pd(a, b, ka="mse", kb="mse"):
    s = sorted(set(a) & set(b)); return [(a[i][ka] / b[i][kb] - 1) * 100 for i in s]
m = lambda d, k="mse": st.mean(x[k] for x in d.values())
fm = lambda x: f"{x:+.2f}%".replace("-", "−")
legacy = collections.defaultdict(dict)
for r in rows:
    if r["_file"] == "main.jsonl" and r["identity"] == "none":
        legacy[(r["data"], r["pred_len"], r["model"])][r["seed"]] = r

if sec == "base":
    print("model|data|H|changed|n none|official none mse/mae|legacy none|pub|vs pub %|vs legacy %")
    gaps = collections.defaultdict(list)
    for model in ("smamba", "itransformer"):
        for data in ORDER:
            for i, H in enumerate(HS):
                k = (data, H, model); ch = changed(model, data, H)
                n = G[k].get("none", {})
                if not n: print(model, data, H, "MISSING"); continue
                pm, pa = PUB[model][data][i]
                gap = (m(n) / pm - 1) * 100; gaps[model].append(gap)
                lg = (m(n) / m(legacy[k]) - 1) * 100 if ch else 0
                if ch or abs(gap) > 2:
                    print(f"{model}|{data}|{H}|{'Y' if ch else ''}|{len(n)}|{m(n):.4f}/{m(n,'mae'):.4f}|{m(legacy[k]):.4f}|{pm:.3f}|{gap:+.1f}|{lg:+.2f}|ep {[r['best_epoch'] for r in n.values()]}/{next(iter(n.values()))['epochs']}")
    for model, g in gaps.items():
        print(model, "mean gap", f"{st.mean(g):+.2f}", "max abs", f"{max(map(abs, g)):.1f}", "within 3%:", sum(abs(x) <= 3 for x in g), "/", len(g))

if sec == "identity":
    print("data|H|model|none|add (d, verdict)|lr10x|drop0.1|drop0.3|shuf none K8|shuf add K2|shuf add K8 vs none|shuf drop0.1 K8")
    for data in ORDER:
        for H in HS:
            for model in ("smamba", "itransformer"):
                k = (data, H, model); g = G[k]; n = g.get("none")
                if not n: print(data, H, model, "no baseline", sorted(g)); continue
                cell = []
                for v, ka in (("add", "mse"), ("add-lr10x", "mse"), ("add-drop0.1", "mse"), ("add-drop0.3", "mse"),
                              ("none-shuf", "mse_orders8"), ("add-shuf", "mse_orders2"), ("add-shuf", "mse_orders8"), ("add-drop0.1-shuf", "mse_orders8")):
                    if v in g and len(g[v]) >= 2 and ka in next(iter(g[v].values())):
                        d = pd(g[v], n, ka, "mse"); cell.append(f"{fm(st.mean(d))} {verdict(d)}")
                    else:
                        cell.append("-")
                print(f"{data}|{H}|{model}|{m(n):.4f}|" + "|".join(cell))

if sec == "select":
    # test metric per variant: shuffled models use K=2 averaging, others the single pass
    tk = lambda v: "mse_orders2" if v.endswith("-shuf") else "mse"
    tka = lambda v: "mae_orders2" if v.endswith("-shuf") else "mae"
    out = collections.defaultdict(list); table = []
    for model in ("itransformer", "smamba"):
        for data in ORDER:
            for i, H in enumerate(HS):
                k = (data, H, model); g = G[k]; n = g.get("none", {})
                if len(n) < 3:
                    continue
                cands = {v: runs for v, runs in g.items() if len(runs) == 3 and (not v.endswith("-shuf") or "mse_orders2" in next(iter(runs.values())))}
                for name, pool in (("all", cands), ("id", {v: r for v, r in cands.items() if v.startswith("add")})):
                    pick = min(pool, key=lambda v: m(pool[v], "val_mse"))
                    d = pd(pool[pick], n, tk(pick), "mse")
                    out[(model, name)].append((data, H, pick, st.mean(d), verdict(d) if pick != "none" else "-"))
                    if name == "all":
                        r = pool[pick]; pm, pa = PUB[model][data][i]
                        table.append((model, data, H, pick, m(n), m(r, tk(pick)), m(r, tka(pick)), st.mean(d), verdict(d) if pick != "none" else "–", pm, pa,
                                      min(cands, key=lambda v: m(cands[v], tk(v)))))
    for (model, name), v in out.items():
        c = collections.Counter(x[4] for x in v); picks = collections.Counter(x[2] for x in v)
        print(f"{model} pool={name}: n={len(v)} mean {st.mean(x[3] for x in v):+.2f}% | verdicts {dict(c)} | worse-than-none {sum(x[3] > 0 for x in v)} | picks {dict(picks)}")
        if name == "all":
            for data in ORDER:
                xs = [x for x in v if x[0] == data]
                if xs: print(f"   {data}: {st.mean(x[3] for x in xs):+.2f}%  " + " ".join(f"{x[1]}:{x[2]}({x[3]:+.1f})" for x in xs))
    with open(sys.argv[4], "w") as f:
        json.dump(table, f)

if sec == "rules":
    tk = lambda v: "mse_orders2" if v.endswith("-shuf") else "mse"
    def rule_eval(model, rule):
        res = []
        for data in ORDER:
            for H in HS:
                k = (data, H, model); g = G[k]; n = g.get("none", {})
                if len(n) < 3: continue
                v = rule(data, H, g)
                if v is None or v not in g or len(g[v]) < 3 or tk(v) not in next(iter(g[v].values())): res.append((data, H, None)); continue
                d = pd(g[v], n, tk(v), "mse"); res.append((data, H, (st.mean(d), verdict(d), m(g[v], tk(v)))))
        return res
    lowlr = lambda model, data, H: get_config(model, data, H)["lr"] <= 1e-4
    rules = {
        "itransformer": {
            "add": lambda d, H, g: "add",
            "add, lr x10 if lr<=1e-4": lambda d, H, g: "add-lr10x" if lowlr("itransformer", d, H) else "add",
            "add + drop0.3": lambda d, H, g: "add-drop0.3" if "add-drop0.3" in g else None,
            "add, lr x10 if lr<=1e-4, drop0.3 on Traffic": lambda d, H, g: "add-drop0.3" if d == "Traffic" else ("add-lr10x" if lowlr("itransformer", d, H) else "add"),
        },
        "smamba": {
            "add": lambda d, H, g: "add",
            "shuffle + add, K=2": lambda d, H, g: "add-shuf",
            "shuffle + none, K=2": lambda d, H, g: "none-shuf",
            "add, lr x10 if lr<=1e-4": lambda d, H, g: "add-lr10x" if lowlr("smamba", d, H) else "add",
        }}
    for model, rs in rules.items():
        print(f"== {model}")
        for name, rule in rs.items():
            res = rule_eval(model, rule)
            ok = [x for x in res if x[2]]
            c = collections.Counter(x[2][1] for x in ok)
            per = {data: st.mean(x[2][0] for x in ok if x[0] == data) for data in ORDER if any(x[0] == data for x in ok)}
            print(f"  {name}: n={len(ok)} mean {st.mean(x[2][0] for x in ok):+.2f}% | {dict(c)} | " + " ".join(f"{d} {v:+.2f}" for d, v in per.items()))

if sec in ("final", "final_md"):
    lowlr = lambda model, data, H: get_config(model, data, H)["lr"] <= 1e-4
    def it_recipe(d, H): return "add-drop0.3" if d == "Traffic" else ("add-lr10x" if lowlr("itransformer", d, H) else "add")
    rows_out = []; wins = collections.Counter(); beats = collections.Counter(); n_cmp = 0
    for data in ORDER:
        for i, H in enumerate(HS):
            gi, gs = G[(data, H, "itransformer")], G[(data, H, "smamba")]
            ps, pi = PUB["smamba"][data][i], PUB["itransformer"][data][i]
            vi = it_recipe(data, H)
            cand = {}
            if len(gi.get("none", {})) == 3: cand["iT"] = (m(gi["none"]), m(gi["none"], "mae"))
            if len(gi.get(vi, {})) == 3: cand["iT+id"] = (m(gi[vi]), m(gi[vi], "mae"))
            if len(gs.get("none", {})) == 3: cand["SM"] = (m(gs["none"]), m(gs["none"], "mae"))
            if len(gs.get("add-shuf", {})) == 3: cand["SM-shuf+id"] = (m(gs["add-shuf"], "mse_orders2"), m(gs["add-shuf"], "mae_orders2"))
            di = pd(gi[vi], gi["none"]) if "iT+id" in cand and "iT" in cand else None
            ds = pd(gs["add-shuf"], gs["none"], "mse_orders2") if "SM-shuf+id" in cand and "SM" in cand else None
            ours = {k: v for k, v in cand.items() if k in ("iT+id", "SM-shuf+id")}
            best_ours = min(ours, key=lambda k: ours[k][0]) if ours else None
            if best_ours:
                n_cmp += 1
                if ours[best_ours][0] < min(ps[0], pi[0]): beats["mse"] += 1
                if ours[best_ours][1] < min(ps[1], pi[1]): beats["mae"] += 1
                wins[best_ours] += 1
            rows_out.append((data, H, ps, pi, cand, vi, di, ds, best_ours))
    if sec == "final":
        for r in rows_out: print(r[0], r[1], r[8], {k: round(v[0], 4) for k, v in r[4].items()}, r[5])
        print("best of ours:", dict(wins), "beats both published:", dict(beats), "of", n_cmp)
    else:
        fmt = lambda c, k: f"{c[k][0]:.3f} / {c[k][1]:.3f}" if k in c else "–"
        print("| 数据集 | H | 论文 S-Mamba | 论文 iTransformer | 本文 iTransformer | iTransformer + 身份 | 本文 S-Mamba | S-Mamba 随机顺序 + 身份（K=2） |")
        print("|---|---|---|---|---|---|---|---|")
        for data, H, ps, pi, c, vi, di, ds, best in rows_out:
            lo = min([ps[0], pi[0]] + [v[0] for v in c.values()])
            b_ = lambda txt, x: f"**{txt}**" if x == lo else txt
            cell = lambda k, d: (b_(fmt(c, k), c[k][0]) if k in c else "–") + (f"（{st.mean(d):+.1f}%）".replace("-", "−") if d else "")
            tag = {"add": "", "add-lr10x": "†", "add-drop0.3": "‡"}[vi]
            print(f"| {data} | {H} | {b_(f'{ps[0]:.3f} / {ps[1]:.3f}', ps[0])} | {b_(f'{pi[0]:.3f} / {pi[1]:.3f}', pi[0])} | {cell('iT', None)} | "
                  f"{cell('iT+id', di)}{tag} | {cell('SM', None)} | {cell('SM-shuf+id', ds)} |")

if sec == "exp9":
    # legacy view (main + idlr + shuffle_main, legacy settings) vs official view for the changed combos
    Lg = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        if r["_file"] in ("main.jsonl", "idlr.jsonl", "shuffle_main.jsonl") and not r.get("tag"):
            k = (r["data"], r["pred_len"], r["model"])
            leg = get_config(k[2], k[0], k[1], "legacy")
            if training(r) == tuple(leg.get(x) for x in TRAINING_KEYS):
                Lg[k][var(r)].setdefault(r["seed"], r)
    name = {"smamba": "S-Mamba", "itransformer": "iTransformer"}
    print("| 骨架 | 数据集 | H | 论文 | 旧设置基线 | 官方设置基线 | 变化 | 身份 `add`：旧 → 官方 | 身份学习率 ×10：旧 → 官方 | 随机顺序 + 身份（K=8）：旧 → 官方 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for model in ("smamba", "itransformer"):
        for data in ORDER:
            for i, H in enumerate(HS):
                if not changed(model, data, H): continue
                k = (data, H, model); o, l = G[k], Lg[k]
                if len(o.get("none", {})) < 3:
                    print(f"| {name[model]} | {data} | {H} | {PUB[model][data][i][0]:.3f} | {m(l['none']):.4f} | 未跑完 | | | | |"); continue
                def ch(v, ka):
                    f = lambda g: fm(st.mean(pd(g[v], g["none"], ka, "mse"))) if v in g and len(g[v]) == 3 else "–"
                    return f"{f(l)} → {f(o)}"
                print(f"| {name[model]} | {data} | {H} | {PUB[model][data][i][0]:.3f} | {m(l['none']):.4f} | {m(o['none']):.4f} | "
                      f"{fm((m(o['none']) / m(l['none']) - 1) * 100)} | {ch('add', 'mse')} | {ch('add-lr10x', 'mse')} | "
                      f"{ch('add-shuf', 'mse_orders8') if model == 'smamba' else '–'} |")

if sec == "exp10":
    print("| 数据集 | H | 骨架 | 无身份 | `add` | dropout 0.1 | dropout 0.3 | 随机顺序 + 身份（K=8） | 随机顺序 + 身份 + dropout 0.1（K=8） |")
    print("|---|---|---|---|---|---|---|---|---|")
    for data in ("Electricity", "Traffic"):
        for H in HS:
            for model in ("itransformer", "smamba"):
                g = G[(data, H, model)]; n = g.get("none", {})
                def c(v, ka="mse"):
                    if v not in g or len(g[v]) < 3: return "–"
                    if len(n) < 3: return f"{m(g[v], ka):.4f}"
                    d = pd(g[v], n, ka, "mse"); s = f"{fm(st.mean(d))} {verdict(d)}"
                    return f"**{s}**" if verdict(d) == "有效" else s
                base = f"{m(n):.4f}" if len(n) == 3 else ("未跑完" if model == "smamba" else "–")
                print(f"| {data} | {H} | {'iTransformer' if model == 'itransformer' else 'S-Mamba'} | {base} | {c('add')} | {c('add-drop0.1')} | {c('add-drop0.3')} | "
                      + (f"{c('add-shuf', 'mse_orders8')} | {c('add-drop0.1-shuf', 'mse_orders8')} |" if model == "smamba" else "– | – |"))
