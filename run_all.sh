#!/usr/bin/env bash
# Full benchmark: 5 models x 5 seeds on NF-BoT-IoT-v2.
# Estimated time on RTX 4090: ~30-60 minutes total.
set -euo pipefail

MODELS=(mlp cnn1d kan fkan cmlp)
SEEDS=(42 1337 7 2024 31415)

mkdir -p results

for m in "${MODELS[@]}"; do
  for s in "${SEEDS[@]}"; do
    echo "=== model=$m seed=$s ==="
    python train.py --model "$m" --seed "$s" "$@"
  done
done

echo
echo "=== aggregating ==="
python benchmark.py

echo
echo "Done. Push results/ back to git."