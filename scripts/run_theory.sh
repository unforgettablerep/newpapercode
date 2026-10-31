#!/usr/bin/env bash
# Experiments that test the derivation in papers/theory.md (newpaper repository), horizon 96
# (set HORIZONS for more), three seeds, official settings. The design follows from the
# derivation: every variable-specific part is a shared part plus a zero-initialised offset
# (D1), the offsets' learning rate is normalised to a learning budget of about 1 (D2, "auto"),
# a variable-specific dropout trains the fallback and a label-free posterior gate mixes the
# forecasts with and without the offsets of drifting variables (D4), and S-Mamba scans in
# random orders (D5).
#   P2 channel norm (FiLM per layer) subsumes the identity: id / cn / cn+id
#   P3 shared + per-variable cycles beat per-variable cycles: full vs full without --rcf-shared
#   P4 the budget rule matters only where the budget is << 1 (Weather, ETTh1): full vs x1
#   P5 posterior gate >= hard gate >= no gate: all three are evaluated in every full run
#   P6 one random scan order ~ K-order averaging: --order-probe in the S-Mamba runs
# Runs already in $OUT are skipped.
set -euo pipefail
OUT=${OUT:-results/theory.jsonl}
ROOT=${ROOT:-./data}
EXTRA=${EXTRA:-"--tf32"}
DATASETS=${DATASETS:-"Electricity Traffic Weather ETTh1"}
HORIZONS=${HORIZONS:-"96"}
SEEDS=${SEEDS:-"0 1 2"}
PARTS="--cn --rcf-cycle auto --vs-dropout 0.1 --drift-gate"
for data in $DATASETS; do
  pv=""; [ "$data" = Traffic ] && pv="--per-var"
  for H in $HORIZONS; do
    for seed in $SEEDS; do
      for model in itransformer smamba; do
        c="python run.py --model $model --data $data --pred-len $H --seed $seed --root-path $ROOT --out $OUT --skip-existing $pv $EXTRA"
        S=""; [ "$model" = smamba ] && S="--scan-order shuffle --eval-orders 2 --order-probe 3"
        $c --identity none                                                        # baseline (fixed order)
        $c --identity add --id-lr-mult auto $S                                    # identity
        $c --identity none --cn --vs-lr-mult auto $S                              # channel norm
        $c --identity add --cn --id-lr-mult auto --vs-lr-mult auto $S             # both (P2)
        $c --identity none $PARTS --rcf-shared --vs-lr-mult auto $S               # derived model
        $c --identity add --id-lr-mult auto $PARTS --rcf-shared --vs-lr-mult auto $S   # + identity
        $c --identity none $PARTS --vs-lr-mult auto $S                            # per-variable cycles (P3)
        if [ "$model" = smamba ]; then
          $c --identity none $PARTS --rcf-shared --vs-lr-mult auto                # fixed scan order (D5)
        fi
        lr=$(python -c "from idembed.configs import get_config; print(get_config('$model', '$data', $H)['lr'])")
        if python -c "import sys; sys.exit(0 if $lr <= 1e-4 else 1)"; then        # budget << 1 (P4)
          $c --identity none $PARTS --rcf-shared $S
        fi
      done
    done
  done
done
python tools/summarize.py "$OUT"
