#!/usr/bin/env bash
# RNK on the synthetic data (rnk/sim.py): every variant x 3 seeds, CPU is enough (about 1 min/run).
# OUT=results/rnk_sim.jsonl  CACHE=cache  SEEDS="0 1 2"  VARIANTS="..."  bash scripts/run_rnk_sim.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results/rnk_sim.jsonl}
CACHE=${CACHE:-cache}
SEEDS=${SEEDS:-"0 1 2"}
VARIANTS=${VARIANTS:-"global id id_drop nke_noid nke kcd_reg kcd_prior kcd_post kcd_oracle rnk rnk_noid rnk_all"}
PY=${PY:-python}
for seed in $SEEDS; do
  for v in $VARIANTS; do
    echo "== sim $v seed $seed"
    $PY -W ignore run_rnk.py --data sim --variant "$v" --seed "$seed" --cache "$CACHE" --out "$OUT" --quiet
  done
done
