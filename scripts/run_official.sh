#!/usr/bin/env bash
# Rerun every configuration whose default settings changed when idembed/configs.py moved to the
# official per-horizon settings (22 S-Mamba and 3 iTransformer (dataset, horizon) pairs; see
# `python -c "from idembed.configs import changed; ..."` or the README). Three seeds, each with:
#   none, add, and add with identity learning rate x10 where the learning rate is <= 1e-4;
#   S-Mamba also with a shuffled scan order (none, add), tested over 2 / 4 / 8 orders.
# The unchanged configurations keep their earlier results (main.jsonl, idlr.jsonl,
# shuffle_main.jsonl); summarise all files together, the summary pairs only runs with
# identical training settings.
set -euo pipefail
OUT=${OUT:-results/official.jsonl}
ROOT=${ROOT:-./data}
EXTRA=${EXTRA:-"--tf32"}
SEEDS=${SEEDS:-"0 1 2"}
DATASETS=${DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather Electricity Traffic"}
for data in $DATASETS; do
  for model in smamba itransformer; do
    for H in $(python -c "from idembed.configs import changed, horizons
print(*[h for h in horizons('$data') if changed('$model', '$data', h)])"); do
      lr=$(python -c "from idembed.configs import get_config; print(get_config('$model', '$data', $H)['lr'])")
      for seed in $SEEDS; do
        common="--model $model --data $data --pred-len $H --seed $seed --root-path $ROOT --out $OUT $EXTRA"
        python run.py $common --identity none
        python run.py $common --identity add
        if python -c "import sys; sys.exit(0 if $lr <= 1e-4 else 1)"; then
          python run.py $common --identity add --id-lr-mult 10
        fi
        if [ "$model" = smamba ]; then
          python run.py $common --identity none --scan-order shuffle --eval-orders 2,4,8
          python run.py $common --identity add --scan-order shuffle --eval-orders 2,4,8
        fi
      done
    done
  done
done
python tools/summarize.py "$OUT"
