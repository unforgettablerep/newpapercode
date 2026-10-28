#!/usr/bin/env bash
# Ablations of the identity embedding (S-Mamba by default), horizon 96, three seeds:
#   form      : full table (add) vs low-rank r=8 / r=32
#   placement : input only vs before every encoder layer
#   dropout   : identity dropout 0.1
#   lr        : 10x learning rate for the identity parameters
# plus the unseen-variable protocol: train on 40% of the variables, test on all, with no
# identity, identity, and identity + identity dropout 0.3.
set -euo pipefail
OUT=${OUT:-results/ablation.jsonl}
ROOT=${ROOT:-./data}
EXTRA=${EXTRA:-"--tf32"}
MODEL=${MODEL:-smamba}
DATASETS=${DATASETS:-"ETTh1 Weather Electricity"}
SEEDS=${SEEDS:-"0 1 2"}
for data in $DATASETS; do
  for seed in $SEEDS; do
    common="--model $MODEL --data $data --pred-len 96 --seed $seed --root-path $ROOT --out $OUT $EXTRA"
    python run.py $common --identity none
    python run.py $common --identity add
    python run.py $common --identity lowrank --id-rank 8
    python run.py $common --identity lowrank --id-rank 32
    python run.py $common --identity add --id-loc every
    python run.py $common --identity add --id-dropout 0.1
    python run.py $common --identity add --id-lr-mult 10
    # unseen variables
    python run.py $common --train-var-ratio 0.4 --identity none
    python run.py $common --train-var-ratio 0.4 --identity add
    python run.py $common --train-var-ratio 0.4 --identity add --id-dropout 0.3
  done
done
python tools/summarize.py "$OUT"
