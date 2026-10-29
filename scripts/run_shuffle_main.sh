#!/usr/bin/env bash
# S-Mamba trained with a shuffled scan order, with and without identity, on every dataset and
# horizon (three seeds); each run is tested in the original order and averaged over 2 / 4 / 8
# random orders. At horizon 96 this beat the fixed-order S-Mamba on Electricity (-4.1%) and
# ETTh1 (-1.3%) but not on Weather (+0.7%). The fixed-order runs are those of run_main.sh;
# summarise together, e.g.
#   python tools/summarize.py <newpaper>/results/4090/main.jsonl results/shuffle_main.jsonl
set -euo pipefail
OUT=${OUT:-results/shuffle_main.jsonl}
ROOT=${ROOT:-./data}
EXTRA=${EXTRA:-"--tf32"}
DATASETS=${DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather Electricity Traffic"}
HORIZONS=${HORIZONS:-"96 192 336 720"}
SEEDS=${SEEDS:-"0 1 2"}
for data in $DATASETS; do
  for H in $HORIZONS; do
    for seed in $SEEDS; do
      for id in none add; do
        python run.py --model smamba --data $data --pred-len $H --seed $seed --root-path $ROOT \
          --out $OUT $EXTRA --identity $id --scan-order shuffle --eval-orders 2,4,8
      done
    done
  done
done
python tools/summarize.py "$OUT"
