"""Summarise results/*.jsonl: mean ± std over seeds and seed-paired comparisons against the
same configuration without identity embedding.

python tools/summarize.py results/main.jsonl [results/ablation.jsonl ...] > summary.md

Verdict (fixed before running): "effective" if the variant is better on every seed and the
mean paired MSE change is at most -1%; "marginal" if better on every seed but by less;
"worse" if worse on every seed; otherwise "unclear".
"""
import collections
import json
import re
import statistics as st
import sys


def variant(r):
    if r["identity"] == "none":
        return "none"
    v = r["identity"] + (f"-r{r['id_rank']}" if r["identity"] == "lowrank" else "")
    if r["id_loc"] != "input":
        v += f"-{r['id_loc']}"
    if r["id_dropout"]:
        v += f"-drop{r['id_dropout']}"
    if r.get("id_lr_mult", 1.0) != 1.0:
        v += f"-lr{r['id_lr_mult']:g}x"
    return v


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def verdict(d):
    better = sum(x < 0 for x in d)
    if better == len(d):
        return "effective" if st.mean(d) <= -1.0 else "marginal"
    return "worse" if better == 0 else "unclear"


def paired(runs, base, metric):
    seeds = sorted(s for s in set(runs) & set(base) if metric in runs[s] and metric in base[s])
    return [(runs[s][metric] / base[s][metric] - 1) * 100 for s in seeds]


def summarize(rows, metric="mse"):
    # (data, H, model, ratio, scan, perm, tag) -> variant -> seed -> row
    groups = collections.defaultdict(dict)
    for r in rows:
        key = (r["data"], r["pred_len"], r["model"], r.get("train_var_ratio", 1.0),
               r.get("scan_order", "fixed"), r.get("var_perm", -1), r.get("tag", ""))
        groups[key].setdefault(variant(r), {})[r["seed"]] = r
    out = []
    for key in sorted(groups):
        data, H, model, ratio, scan, perm, tag = key
        title = (f"### {data}, H={H}, {model}" + (f", scan order: {scan}" if scan != "fixed" else "")
                 + (f", variables in random order {perm}" if perm != -1 else "")
                 + (f", tag `{tag}`" if tag else "")
                 + (f", trained on {ratio:.0%} of variables" if ratio < 1 else ""))
        present = {c for v in groups[key].values() for r in v.values() for c in r}
        sweep = sorted((c for c in present if re.fullmatch(r"mse_orders\d+", c)),
                       key=lambda c: int(c[10:]))
        has_orders = "mse_orders" in present
        cols = (["mse", "mae"] + (sweep or (["mse_orders", "mae_orders"] if has_orders else []))
                + (["mse_seen", "mse_unseen"] if ratio < 1 else []))
        out += [title, "", "| variant | " + " | ".join(cols) + " | seeds | vs none (paired MSE) | verdict |",
                "|---|" + "---|" * (len(cols) + 3)]
        base = groups[key].get("none", {})
        for v in sorted(groups[key], key=lambda s: (s != "none", s)):
            runs = groups[key][v]
            cells = []
            for c in cols:
                xs = [r[c] for r in runs.values() if c in r]
                cells.append(f"{st.mean(xs):.4f} ± {st.stdev(xs):.4f}" if len(xs) > 1 else
                             (f"{xs[0]:.4f}" if xs else "–"))
            d = paired(runs, base, metric) if v != "none" else []
            if len(d) >= 2:
                cmp_, ver = f"{st.mean(d):+.2f}% ({sum(x < 0 for x in d)}/{len(d)} better)", verdict(d)
            else:
                cmp_, ver = "–", "–"
            out.append(f"| {v} | " + " | ".join(cells) + f" | {len(runs)} | {cmp_} | {ver} |")
        if has_orders:
            out.append("")
            out.append("(`*_orders`, `mse_ordersK`: the same trained model tested with the mixing "
                       "averaged over several / K random scan orders)")
        out.append("")
    out += order_hypothesis(groups, metric)
    return "\n".join(out)


def order_hypothesis(groups, metric):
    """Identity gain with the original scan order vs with a shuffled scan order (S-Mamba)."""
    lines = []
    for (data, H, model, ratio, scan, perm, tag) in sorted(groups):
        if scan != "shuffle" or (data, H, model, ratio, "fixed", perm, tag) not in groups:
            continue
        fixed = groups[(data, H, model, ratio, "fixed", perm, tag)]
        shuf = groups[(data, H, model, ratio, scan, perm, tag)]
        if not all(k in g for g in (fixed, shuf) for k in ("none", "add")):
            continue
        g_fixed = paired(fixed["add"], fixed["none"], metric)
        g_shuf = paired(shuf["add"], shuf["none"], metric)
        cost = paired(shuf["none"], fixed["none"], metric)
        if lines == []:
            lines = ["### Does the fixed scan order act as an identity? (S-Mamba, identity `add`)", "",
                     "| data | H | identity gain, fixed order | identity gain, shuffled order "
                     "| shuffling cost without identity |", "|---|---|---|---|---|"]
        f = lambda d: f"{st.mean(d):+.2f}% ({sum(x < 0 for x in d)}/{len(d)})" if d else "–"
        lines.append(f"| {data} | {H} | {f(g_fixed)} | {f(g_shuf)} | {f(cost)} |")
    return lines + [""] if lines else []


if __name__ == "__main__":
    print(summarize(load(sys.argv[1:])))
