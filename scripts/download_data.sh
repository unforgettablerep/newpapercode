#!/usr/bin/env bash
# Download the long-term forecasting benchmarks from the Time-Series-Library mirror on
# Hugging Face into ./data (same layout the loaders expect).
# PEMS03/04/07/08 are not in that mirror: put PEMS0X.npz into ./data/PEMS/ yourself
# (the iTransformer / S-Mamba dataset release).
set -euo pipefail
ROOT=${1:-./data}
BASE=https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main
FILES=(
  ETT-small/ETTh1.csv ETT-small/ETTh2.csv ETT-small/ETTm1.csv ETT-small/ETTm2.csv
  weather/weather.csv electricity/electricity.csv traffic/traffic.csv
  exchange_rate/exchange_rate.csv
)
for f in "${FILES[@]}"; do
  mkdir -p "$ROOT/$(dirname "$f")"
  if [ -s "$ROOT/$f" ]; then echo "exists  $f"; continue; fi
  echo "fetch   $f"
  curl -fL --retry 3 -o "$ROOT/$f" "$BASE/$f"
done
mkdir -p "$ROOT/PEMS"
echo "done. PEMS files present: $(ls "$ROOT/PEMS" 2>/dev/null | tr '\n' ' ')"
