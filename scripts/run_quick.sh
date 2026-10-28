#!/usr/bin/env bash
# Quick check of the main claim at full model size: identity embeddings on top of S-Mamba and
# iTransformer, horizon 96, three datasets, three seeds (36 runs).
set -euo pipefail
OUT=${OUT:-results/quick.jsonl}
ROOT=${ROOT:-./data}
EXTRA=${EXTRA:-"--tf32"}
for data in ETTh1 Weather Electricity; do
  for model in smamba itransformer; do
    for id in none add; do
      for seed in 0 1 2; do
        python run.py --model $model --data $data --pred-len 96 --identity $id --seed $seed \
          --root-path "$ROOT" --out "$OUT" $EXTRA
      done
    done
  done
done
python tools/summarize.py "$OUT"
