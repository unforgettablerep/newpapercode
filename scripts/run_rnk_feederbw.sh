#!/usr/bin/env bash
# RNK on FeederBW (rnk/feederbw.py): temporal split (all feeders of the given packs) and
# locational split (train on the first half of the feeders, test on the other half), 3 seeds.
# ZIPS="feeder_measurement_data_161-200.zip" WEATHER=weather_data.parquet META=feeder_metadata.csv \
#   OUT=results/rnk_feederbw.jsonl CACHE=cache bash scripts/run_rnk_feederbw.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results/rnk_feederbw.jsonl}
CACHE=${CACHE:-cache}
SEEDS=${SEEDS:-"0 1 2"}
TEMPORAL=${TEMPORAL:-"global id id_drop nke_noid nke kcd_reg kcd_prior kcd_post rnk rnk_noid"}
LOCATIONAL=${LOCATIONAL:-"global nke_noid kcd_reg kcd_post rnk_noid"}
PY=${PY:-python}
common=(--data feederbw --zips $ZIPS --weather "$WEATHER" --meta "$META" --cache "$CACHE" --out "$OUT" --quiet)
for seed in $SEEDS; do
  for v in $TEMPORAL; do
    echo "== feederbw temporal $v seed $seed"
    $PY -W ignore run_rnk.py "${common[@]}" --variant "$v" --seed "$seed" --split temporal
  done
  for v in $LOCATIONAL; do
    echo "== feederbw locational $v seed $seed"
    $PY -W ignore run_rnk.py "${common[@]}" --variant "$v" --seed "$seed" --split locational
  done
done
