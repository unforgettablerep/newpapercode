#!/usr/bin/env bash
# Identity dropout on the two datasets with many variables, every horizon, three seeds.
# On Traffic (horizon 96) it turned the identity from harmful into helpful for iTransformer
# (-1.3% / -2.4% MSE with dropout 0.1 / 0.3) and helped S-Mamba slightly; on Electricity it
# was the only ablation variant that was effective in both runs.
#   iTransformer and S-Mamba, fixed scan order: add + dropout 0.1, add + dropout 0.3
#   S-Mamba, shuffled scan order: add + dropout 0.1, tested over 2 / 8 orders
# Traffic runs store per-variable errors. The runs without identity / without dropout are
# those of run_main.sh (iTransformer, S-Mamba Electricity-96) and run_official.sh (the other
# S-Mamba configurations); summarise all files together.
set -euo pipefail
OUT=${OUT:-results/dropout.jsonl}
ROOT=${ROOT:-./data}
EXTRA=${EXTRA:-"--tf32"}
DATASETS=${DATASETS:-"Electricity Traffic"}
HORIZONS=${HORIZONS:-"96 192 336 720"}
SEEDS=${SEEDS:-"0 1 2"}
for data in $DATASETS; do
  pv=""; [ "$data" = Traffic ] && pv="--per-var"
  for H in $HORIZONS; do
    for seed in $SEEDS; do
      for model in itransformer smamba; do
        common="--model $model --data $data --pred-len $H --seed $seed --root-path $ROOT --out $OUT $pv $EXTRA"
        python run.py $common --identity add --id-dropout 0.1
        python run.py $common --identity add --id-dropout 0.3
        if [ "$model" = smamba ]; then
          python run.py $common --identity add --id-dropout 0.1 --scan-order shuffle --eval-orders 2,8
        fi
      done
    done
  done
done
python tools/summarize.py "$OUT"
