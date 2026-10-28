#!/usr/bin/env bash
# Main table: every dataset x horizon x backbone, with and without identity embeddings,
# three seeds. Datasets whose files are missing are skipped.
# Usage: DATASETS="Electricity Traffic" SEEDS="0 1 2" bash scripts/run_main.sh
set -euo pipefail
OUT=${OUT:-results/main.jsonl}
ROOT=${ROOT:-./data}
EXTRA=${EXTRA:-"--tf32"}
DATASETS=${DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange Electricity Traffic PEMS03 PEMS04 PEMS07 PEMS08"}
MODELS=${MODELS:-"smamba itransformer"}
SEEDS=${SEEDS:-"0 1 2"}
for data in $DATASETS; do
  file=$(python -c "from idembed.data import DATASETS; print(DATASETS['$data'][0])")
  if [ ! -f "$ROOT/$file" ]; then echo "skip $data (missing $ROOT/$file)"; continue; fi
  for H in $(python -c "from idembed.configs import horizons; print(*horizons('$data'))"); do
    for model in $MODELS; do
      for id in none add; do
        for seed in $SEEDS; do
          python run.py --model $model --data $data --pred-len $H --identity $id --seed $seed \
            --root-path "$ROOT" --out "$OUT" $EXTRA
        done
      done
    done
  done
done
python tools/summarize.py "$OUT"
