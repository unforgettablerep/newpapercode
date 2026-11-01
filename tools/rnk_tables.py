"""Markdown tables of selected (variant, mode) rows for the RNK write-up: mean over seeds,
seed standard deviation, and the change relative to a reference row.

  python tools/rnk_tables.py results/rnk_sim.jsonl --rows global:trust,nke:trust,rnk:gate_err \
      --masks seen_stable,seen_undoc_0_60 [--metric mse] [--ref global:trust] [--split temporal]
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def collect(paths, metric, split=None):
    acc = defaultdict(dict)
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            c = r["cfg"]
            if split and c.get("split", "temporal") != split:
                continue
            for mode, mm in r["metrics"].items():
                for mask, v in mm.items():
                    acc[(c["variant"], mode, mask)][c["seed"]] = v[metric]
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--metric", default="mse")
    ap.add_argument("--ref", default="global:trust")
    ap.add_argument("--split")
    ap.add_argument("--digits", type=int, default=4)
    a = ap.parse_args()
    acc = collect(a.paths, a.metric, a.split)
    rows = [tuple(r.split(":")) for r in a.rows.split(",")]
    masks = a.masks.split(",")
    ref = tuple(a.ref.split(":"))
    print("| variant | mode | " + " | ".join(masks) + " |")
    print("|---|---|" + "---|" * len(masks))
    for v, mode in rows:
        cells = []
        for mk in masks:
            d = acc.get((v, mode, mk), {})
            if not d:
                cells.append("–")
                continue
            vals = np.array(list(d.values()))
            s = f"{vals.mean():.{a.digits}f}"
            if len(vals) > 1:
                s += f" ± {vals.std(ddof=1):.{a.digits}f}"
            rd = acc.get((ref[0], ref[1], mk), {})
            common = sorted(set(d) & set(rd))
            if common and (v, mode) != ref:
                ch = np.mean([d[k] / rd[k] - 1 for k in common]) * 100
                better = sum(d[k] < rd[k] for k in common)
                s += f" ({ch:+.1f}%, {better}/{len(common)})"
            cells.append(s)
        print(f"| {v} | {mode} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
