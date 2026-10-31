#!/usr/bin/env bash
# Candidate final recipes on the six datasets where they have not been run yet (ETT*,
# Exchange, Weather), every horizon, three seeds, official settings:
#   iTransformer: add + identity dropout 0.3, with identity learning rate x1 and x10
#   S-Mamba: shuffled scan order + add + identity dropout 0.1, tested over 2 / 8 orders
# Together with run_dropout.sh (Electricity, Traffic) this gives each recipe on all eight
# datasets. The shuffled runs also store the validation MSE averaged over the same orders
# (val_mse_orders2 / 8), so shuffled and fixed-order models can be compared on validation.
# Runs already in $OUT are skipped.
set -euo pipefail
OUT=${OUT:-results/recipe.jsonl}
ROOT=${ROOT:-./data}
EXTRA=${EXTRA:-"--tf32"}
DATASETS=${DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather"}
HORIZONS=${HORIZONS:-"96 192 336 720"}
SEEDS=${SEEDS:-"0 1 2"}
for data in $DATASETS; do
  for H in $HORIZONS; do
    for seed in $SEEDS; do
      common="--pred-len $H --seed $seed --root-path $ROOT --out $OUT --skip-existing $EXTRA"
      python run.py --model itransformer --data $data $common --identity add --id-dropout 0.3
      python run.py --model itransformer --data $data $common --identity add --id-dropout 0.3 --id-lr-mult 10
      python run.py --model smamba --data $data $common --identity add --id-dropout 0.1 \
        --scan-order shuffle --eval-orders 2,8
    done
  done
done
python tools/summarize.py "$OUT"
