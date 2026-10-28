"""Summarise results/*.jsonl: mean ± std over seeds and seed-paired comparisons against the
same configuration without identity embedding.

python tools/summarize.py results/main.jsonl [results/ablation.jsonl ...] > summary.md

Verdict (fixed before running): "effective" if the variant is better on every seed and the
mean paired MSE change is at most -1%; "marginal" if better on every seed but by less;
"worse" if worse on every seed; otherwise "unclear".
"""
import collections
import json
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


def summarize(rows, metric="mse"):
    groups = collections.defaultdict(dict)        # (data, H, model, ratio) -> variant -> seed -> row
    for r in rows:
        key = (r["data"], r["pred_len"], r["model"], r.get("train_var_ratio", 1.0))
        groups[key].setdefault(variant(r), {})[r["seed"]] = r
    out = []
    for key in sorted(groups):
        data, H, model, ratio = key
        title = f"### {data}, H={H}, {model}" + (f", trained on {ratio:.0%} of variables" if ratio < 1 else "")
        cols = ["mse", "mae"] + (["mse_seen", "mse_unseen"] if ratio < 1 else [])
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
            seeds = sorted(set(runs) & set(base))
            if v != "none" and len(seeds) >= 2:
                d = [(runs[s][metric] / base[s][metric] - 1) * 100 for s in seeds]
                cmp_, ver = f"{st.mean(d):+.2f}% ({sum(x < 0 for x in d)}/{len(d)} better)", verdict(d)
            else:
                cmp_, ver = "–", "–"
            out.append(f"| {v} | " + " | ".join(cells) + f" | {len(runs)} | {cmp_} | {ver} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print(summarize(load(sys.argv[1:])))
