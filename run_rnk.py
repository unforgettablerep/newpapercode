"""Train and test one RNK variant; append the result as one JSON line (rnk/experiment.py).

Examples
  python run_rnk.py --data sim --variant rnk --seed 0 --out results/rnk_sim.jsonl
  python run_rnk.py --data feederbw --variant kcd_post --split temporal --seed 0 \
      --zips feeder_measurement_data_161-200.zip --weather weather_data.parquet \
      --meta feeder_metadata.csv --cache cache/ --out results/rnk_feederbw.jsonl
"""
import argparse
import json
import os

from rnk.experiment import VARIANTS, run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=["sim", "feederbw"], required=True)
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--split", choices=["temporal", "locational"], default="temporal")
    ap.add_argument("--zips", nargs="*", default=[])
    ap.add_argument("--weather")
    ap.add_argument("--meta")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/rnk.jsonl")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cfg = {k.replace("-", "_"): v for k, v in vars(a).items() if k not in ("out", "quiet")}
    res = run(cfg, log=(lambda *x: None) if a.quiet else print)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    m = res["metrics"]
    for mode in m:
        part = {k: round(v["mse"], 4) for k, v in m[mode].items() if not k.endswith("_sunny")}
        print(mode, json.dumps(part))


if __name__ == "__main__":
    main()
