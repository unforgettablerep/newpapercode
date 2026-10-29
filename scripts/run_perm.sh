#!/usr/bin/env bash
# Is the file's column order itself useful to S-Mamba? With a shuffled scan order, identity
# recovers the loss on Electricity but not on Weather, whose columns list related quantities
# next to each other. Two checks, horizon 96, three seeds:
#   1. S-Mamba with a fixed scan order over the variables listed in a fixed random order
#      (--var-perm), with and without identity; the file-order runs are those of run_quick.sh.
#   2. Shuffled scan order with 10 epochs instead of 5 (Weather and Electricity; the shuffled
#      runs with identity of run_order.sh were still improving at the 5-epoch cap).
# Summarise together with the earlier files, e.g.
#   python tools/summarize.py <newpaper>/results/4090/quick_T96.jsonl <newpaper>/results/4090/order.jsonl results/perm.jsonl
set -euo pipefail
OUT=${OUT:-results/perm.jsonl}
ROOT=${ROOT:-./data}
EXTRA=${EXTRA:-"--tf32"}
SEEDS=${SEEDS:-"0 1 2"}
PERMS=${PERMS:-"1"}
for seed in $SEEDS; do
  for data in ETTh1 Weather Electricity; do
    common="--model smamba --data $data --pred-len 96 --seed $seed --root-path $ROOT --out $OUT $EXTRA"
    for p in $PERMS; do
      python run.py $common --identity none --var-perm $p
      python run.py $common --identity add --var-perm $p
    done
  done
  for data in Weather Electricity; do
    common="--model smamba --data $data --pred-len 96 --seed $seed --root-path $ROOT --out $OUT $EXTRA"
    for id in none add; do
      python run.py $common --identity $id --scan-order shuffle --eval-orders 4 --epochs 10 --tag epochs10
    done
  done
done
python tools/summarize.py "$OUT"
