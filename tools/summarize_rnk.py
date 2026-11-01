"""Tables for RNK results (run_rnk.py JSON lines): mean MSE over seeds per variant / mode / mask,
and the change relative to the global model.

  python tools/summarize_rnk.py results/rnk_sim.jsonl [--metric mse|mae|mae_kw] [--masks a,b,...]
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return rows


def table(rows, metric="mse", masks=None, modes=None, base=("global", "trust"), split=None):
    acc = defaultdict(list)
    seeds = defaultdict(set)
    for r in rows:
        c = r["cfg"]
        if split and c.get("split", "temporal") != split:
            continue
        for mode, mm in r["metrics"].items():
            for mask, v in mm.items():
                acc[(c["variant"], mode, mask)].append(v[metric])
                seeds[(c["variant"], mode)].add(c["seed"])
    all_masks = sorted({k[2] for k in acc})
    masks = masks or [m for m in all_masks if not m.endswith("_sunny")] + [m for m in all_masks if m.endswith("_sunny")]
    keys = sorted({(k[0], k[1]) for k in acc}, key=lambda t: (t[0], t[1]))
    if modes:
        keys = [k for k in keys if k[1] in modes]
    lines = ["| variant | mode | seeds | " + " | ".join(masks) + " |",
             "|---|---|---|" + "---|" * len(masks)]
    for v, mode in keys:
        cells = []
        for mk in masks:
            a = acc.get((v, mode, mk))
            b = acc.get((base[0], base[1], mk))
            if not a:
                cells.append("–")
                continue
            s = f"{np.mean(a):.4f}"
            if b and (v, mode) != base:
                s += f" ({(np.mean(a) / np.mean(b) - 1) * 100:+.1f}%)"
            cells.append(s)
        lines.append(f"| {v} | {mode} | {len(seeds[(v, mode)])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--metric", default="mse")
    ap.add_argument("--masks")
    ap.add_argument("--modes")
    ap.add_argument("--split")
    a = ap.parse_args()
    rows = load(a.paths)
    print(table(rows, a.metric, a.masks.split(",") if a.masks else None,
                a.modes.split(",") if a.modes else None, split=a.split))


if __name__ == "__main__":
    main()
