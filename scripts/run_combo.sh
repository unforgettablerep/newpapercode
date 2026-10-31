#!/usr/bin/env bash
# Screening of the combined model at horizon 96 (set HORIZONS for more), three seeds, official
# settings. Each component targets one property of multi-node load / traffic data:
#   identity (add)       heterogeneous variables
#   channel norm (--cn)  heterogeneous scales and dynamics of the variables
#   cycles (--rcf-cycle) strong daily / weekly periodicity (CycleNet's residual cycles)
#   --vs-dropout 0.1 + --drift-gate   variables whose behaviour drifts at test time
#   S-Mamba: shuffled scan order + 2-order averaging   no natural order of the variables
# Runs per backbone: the baseline, the full model, and the full model without each component.
# "Without the drift gate" needs no run: every gated run also reports its ungated MSE.
# Where the learning rate is <= 1e-4 (Weather, ETTh1) the full model is also run with 10x
# learning rate for the variable-specific parameters. Runs already in $OUT are skipped.
set -euo pipefail
OUT=${OUT:-results/combo.jsonl}
ROOT=${ROOT:-./data}
EXTRA=${EXTRA:-"--tf32"}
DATASETS=${DATASETS:-"Electricity Traffic Weather ETTh1"}
HORIZONS=${HORIZONS:-"96"}
SEEDS=${SEEDS:-"0 1 2"}
ID="--identity add"; CN="--cn"; RCF="--rcf-cycle auto"; VS="--vs-dropout 0.1 --drift-gate"
SHUF="--scan-order shuffle --eval-orders 2"
for data in $DATASETS; do
  pv=""; [ "$data" = Traffic ] && pv="--per-var"
  for H in $HORIZONS; do
    for seed in $SEEDS; do
      for model in itransformer smamba; do
        c="python run.py --model $model --data $data --pred-len $H --seed $seed --root-path $ROOT --out $OUT --skip-existing $pv $EXTRA"
        S=""; [ "$model" = smamba ] && S="$SHUF"
        $c --identity none                                  # baseline (fixed scan order)
        $c $ID $CN $RCF $VS $S                              # full model
        $c --identity none $CN $RCF $VS $S                  # without identity
        $c $ID $RCF $VS $S                                  # without channel norm
        $c $ID $CN $VS $S                                   # without cycles
        $c $ID $CN $RCF --drift-gate $S                     # without variable-specific dropout
        if [ "$model" = smamba ]; then $c $ID $CN $RCF $VS; fi   # without the shuffled order
        lr=$(python -c "from idembed.configs import get_config; print(get_config('$model', '$data', $H)['lr'])")
        if python -c "import sys; sys.exit(0 if $lr <= 1e-4 else 1)"; then
          $c $ID $CN $RCF $VS $S --id-lr-mult 10 --vs-lr-mult 10
        fi
      done
    done
  done
done
python tools/summarize.py "$OUT"
