#!/usr/bin/env bash
# Why does the identity embedding hurt on Traffic? In the main table its test MSE is 1.5-6.9%
# worse on every seed and horizon, while its validation MSE is better and its MAE about the
# same. Horizon 96 (set HORIZONS for more), three seeds, per-variable errors stored (--per-var)
# for tools/per_variable.py:
#   S-Mamba and iTransformer: none, add, add + identity dropout 0.1 / 0.3, low-rank r=8
#   S-Mamba, 10 epochs instead of 5 (every main-table run stopped at the cap): none, add
#   S-Mamba, shuffled scan order, tested in 1 / 4 / 8 orders: none, add
set -euo pipefail
OUT=${OUT:-results/traffic.jsonl}
ROOT=${ROOT:-./data}
EXTRA=${EXTRA:-"--tf32"}
HORIZONS=${HORIZONS:-"96"}
SEEDS=${SEEDS:-"0 1 2"}
for H in $HORIZONS; do
  for seed in $SEEDS; do
    for model in smamba itransformer; do
      common="--model $model --data Traffic --pred-len $H --seed $seed --root-path $ROOT --out $OUT --per-var $EXTRA"
      python run.py $common --identity none
      python run.py $common --identity add
      python run.py $common --identity add --id-dropout 0.1
      python run.py $common --identity add --id-dropout 0.3
      python run.py $common --identity lowrank --id-rank 8
    done
    common="--model smamba --data Traffic --pred-len $H --seed $seed --root-path $ROOT --out $OUT --per-var $EXTRA"
    python run.py $common --identity none --epochs 10 --tag epochs10
    python run.py $common --identity add --epochs 10 --tag epochs10
    python run.py $common --identity none --scan-order shuffle --eval-orders 4,8
    python run.py $common --identity add --scan-order shuffle --eval-orders 4,8
  done
done
python tools/summarize.py "$OUT"
python tools/per_variable.py "$OUT"
