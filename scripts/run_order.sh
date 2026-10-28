#!/usr/bin/env bash
# Does S-Mamba's fixed scan order act as an implicit variable identity?
# S-Mamba trained with a random scan order at every step, with and without identity
# embeddings; each run is tested in the original order and averaged over 4 random orders.
# The fixed-order counterparts are the S-Mamba runs of run_quick.sh (same seeds); set
# RUN_FIXED=1 to train them here as well. Summarise both files together, e.g.
#   python tools/summarize.py results/quick.jsonl results/order.jsonl
set -euo pipefail
OUT=${OUT:-results/order.jsonl}
ROOT=${ROOT:-./data}
EXTRA=${EXTRA:-"--tf32"}
DATASETS=${DATASETS:-"ETTh1 Weather Electricity"}
SEEDS=${SEEDS:-"0 1 2"}
for data in $DATASETS; do
  for seed in $SEEDS; do
    common="--model smamba --data $data --pred-len 96 --seed $seed --root-path $ROOT --out $OUT $EXTRA"
    for id in none add; do
      python run.py $common --identity $id --scan-order shuffle --eval-orders 4
      if [ "${RUN_FIXED:-0}" = 1 ]; then python run.py $common --identity $id; fi
    done
  done
done
python tools/summarize.py "$OUT"
