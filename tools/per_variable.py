"""Where does the identity embedding help or hurt? Uses the per-variable errors of runs made
with --per-var, paired with the run without identity of the same configuration and seed.

python tools/per_variable.py results/traffic.jsonl [more.jsonl ...] [--top 10]

For every configuration and identity variant it reports, averaged over the seeds:
  - the change of the test MSE, and the same change with the 1% / 5% / 10% most harmed
    variables left out (is the change spread over all variables or due to a few?);
  - how many variables get better / worse, and whether the same variables are harmed in
    every seed (mean Spearman correlation of the per-variable changes between seeds);
  - the Spearman correlation of a variable's test change with its validation change, and
    with how far its test split has moved from the training split: level shift |mean| and
    scale shift |log std| in units of the training statistics, and the level shift from the
    validation to the test split;
  - the most harmed variables.
Per-variable changes are relative (MSE with identity / MSE without - 1).
"""
import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import by_config, training, variant  # noqa: E402


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def without_worst(base, new, frac):
    """Relative change of the mean MSE when the `frac` most harmed variables are left out."""
    k = int(round(len(base) * frac))
    keep = np.argsort(new - base)[: len(base) - k]
    return (new[keep].mean() / base[keep].mean() - 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    groups = collections.defaultdict(dict)   # config -> variant -> seed -> per_var
    for f in a.files:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if "per_var" not in r:
                continue
            key = (r["data"], r["pred_len"], r["model"], r.get("scan_order", "fixed"),
                   r.get("var_perm", -1), r.get("epochs"), r.get("tag", ""), training(r))
            groups[key].setdefault(variant(r), {})[r["seed"]] = r["per_var"]
    for key in sorted(groups, key=by_config):
        data, H, model, scan, perm, epochs, tag, _ = key
        base = groups[key].get("none")
        if not base:
            continue
        for v in sorted(x for x in groups[key] if x != "none"):
            seeds = sorted(set(groups[key][v]) & set(base))
            if not seeds:
                continue
            stats, rel = collections.defaultdict(list), []
            for s in seeds:
                b, n = base[s], groups[key][v][s]
                bt, nt = np.array(b["test_mse"]), np.array(n["test_mse"])
                d = (nt + 1e-12) / (bt + 1e-12) - 1
                rel.append(d)
                stats["total"].append((nt.mean() / bt.mean() - 1) * 100)
                for frac in (0.01, 0.05, 0.10):
                    stats[f"w/o {frac:.0%}"].append(without_worst(bt, nt, frac))
                stats["worse"].append(int((d > 0).sum()))
                if "val_mse" in b and "val_mse" in n:
                    dv = np.array(n["val_mse"]) / np.array(b["val_mse"]) - 1
                    stats["vs val change"].append(spearman(d, dv))
                level, scale = np.abs(b["test_mean"]), np.abs(np.log(b["test_std"]))
                stats["vs level shift"].append(spearman(d, level))
                stats["vs scale shift"].append(spearman(d, scale))
                stats["vs val->test level shift"].append(
                    spearman(d, np.abs(np.array(b["test_mean"]) - np.array(b["val_mean"]))))
            nv = len(rel[0])
            title = (f"### {data}, H={H}, {model}, {v}" + (f", scan {scan}" if scan != "fixed" else "")
                     + (f", variables in random order {perm}" if perm != -1 else "")
                     + (f", tag `{tag}`" if tag else "")
                     + f", epochs {epochs}, seeds {seeds}, {nv} variables")
            print(title + "\n")
            m = lambda k: st.mean(stats[k])
            print(f"- test MSE change {m('total'):+.2f}%; leaving out the most harmed 1% / 5% / 10%: "
                  f"{m('w/o 1%'):+.2f}% / {m('w/o 5%'):+.2f}% / {m('w/o 10%'):+.2f}%")
            print(f"- variables worse: {m('worse'):.0f} of {nv}"
                  + (f"; same variables harmed across seeds: Spearman "
                     f"{st.mean(spearman(x, y) for i, x in enumerate(rel) for y in rel[i + 1:]):+.2f}"
                     if len(rel) > 1 else ""))
            print("- Spearman of the per-variable test change with: "
                  + "; ".join(f"{k[3:]} {m(k):+.2f}" for k in stats if k.startswith("vs ")))
            mean_d = np.mean(rel, 0)
            cols, b0 = base[seeds[0]]["columns"], base[seeds[0]]
            print("- most harmed (mean over seeds): " + ", ".join(
                f"{cols[i]} {mean_d[i] * 100:+.1f}% (level {b0['test_mean'][i]:+.2f}, "
                f"std {b0['test_std'][i]:.2f})" for i in np.argsort(-mean_d)[: a.top]))
            print()


if __name__ == "__main__":
    main()
