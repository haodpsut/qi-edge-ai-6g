#!/usr/bin/env bash
# Full benchmark: 5 models x 5 seeds on NF-BoT-IoT-v2.
# All stdout/stderr is teed to logs/run_all_<timestamp>.log so the run is
# fully reproducible and the log can be pushed back to git for review.
set -euo pipefail

MODELS=(mlp cnn1d kan fkan cmlp)
SEEDS=(42 1337 7 2024 31415)

mkdir -p results logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/run_all_${TS}.log"

echo "Logging to $LOG"

{
  echo "=== run_all.sh started at $(date) ==="
  echo "host:    $(hostname)"
  echo "user:    $(whoami)"
  echo "cwd:     $(pwd)"
  echo "python:  $(python --version)"
  echo "torch:   $(python -c 'import torch; print(torch.__version__, "cuda=" + str(torch.cuda.is_available()), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
  echo "git:     $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo "models:  ${MODELS[*]}"
  echo "seeds:   ${SEEDS[*]}"
  echo "=================================="

  for m in "${MODELS[@]}"; do
    for s in "${SEEDS[@]}"; do
      echo
      echo "=== model=$m seed=$s @ $(date +%H:%M:%S) ==="
      python train.py --model "$m" --seed "$s" "$@"
    done
  done

  echo
  echo "=== aggregating ==="
  python benchmark.py

  echo
  echo "=== run_all.sh finished at $(date) ==="
} 2>&1 | tee "$LOG"

echo
echo "Done. Log saved: $LOG"
echo "Push results + log back so Claude can pull and analyse:"
echo "  git add results/ logs/"
echo "  git commit -m 'benchmark results from RTX 4090'"
echo "  git push"
