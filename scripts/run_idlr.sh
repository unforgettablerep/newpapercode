#!/usr/bin/env bash
# Identity learning rate x10 on every dataset whose learning rate is at most 1e-4 (ETT*,
# Exchange, Weather; both models), all horizons, three seeds. On Weather (S-Mamba, horizon 96)
# this turned the identity gain from -1.8% into -5.9%. The runs without identity are those of
# run_main.sh with the same seeds; summarise together, e.g.
#   python tools/summarize.py <newpaper>/results/4090/main.jsonl results/idlr.jsonl
set -euo pipefail
OUT=${OUT:-results/idlr.jsonl}
ROOT=${ROOT:-./data}
EXTRA=${EXTRA:-"--tf32"}
DATASETS=${DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather"}
MODELS=${MODELS:-"smamba itransformer"}
HORIZONS=${HORIZONS:-"96 192 336 720"}
SEEDS=${SEEDS:-"0 1 2"}
for data in $DATASETS; do
  for H in $HORIZONS; do
    for model in $MODELS; do
      for seed in $SEEDS; do
        python run.py --model $model --data $data --pred-len $H --seed $seed --root-path $ROOT \
          --out $OUT $EXTRA --identity add --id-lr-mult 10
      done
    done
  done
done
python tools/summarize.py "$OUT"
