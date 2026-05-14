#!/usr/bin/env bash
# Download NF-BoT-IoT-v2 (Sarhan NetFlow v2) from Kaggle.
# Source: https://www.kaggle.com/datasets/dhoogla/nfbotiotv2
#
# Prerequisites:
#   1. pip install kaggle (already in environment.yml)
#   2. Kaggle API token at ~/.kaggle/kaggle.json
#      Get it at https://www.kaggle.com/settings/account → "Create New Token"
#      chmod 600 ~/.kaggle/kaggle.json
#
# Output:
#   data/raw/NF-BoT-IoT-v2.parquet  (or .csv depending on version)

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw

if ! command -v kaggle &> /dev/null; then
  echo "ERROR: kaggle CLI not found."
  echo "Install:  pip install kaggle"
  echo "Token:    ~/.kaggle/kaggle.json from https://www.kaggle.com/settings"
  echo ""
  echo "Manual fallback:"
  echo "  1. Go to https://www.kaggle.com/datasets/dhoogla/nfbotiotv2"
  echo "  2. Download + unzip into data/raw/"
  exit 1
fi

if [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "ERROR: ~/.kaggle/kaggle.json missing."
  echo "Get token at https://www.kaggle.com/settings/account"
  echo "Then: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi

echo "Downloading NF-BoT-IoT-v2 from Kaggle (dhoogla/nfbotiotv2)..."
kaggle datasets download -d dhoogla/nfbotiotv2 -p data/raw --unzip

echo ""
echo "Files in data/raw/:"
ls -lh data/raw/

echo ""
echo "Done. Verify with:"
echo "  python -c \"from data import load_nf_bot_iot_v2; X,_,_,_,d=load_nf_bot_iot_v2('data/raw/NF-BoT-IoT-V2.parquet', n_samples=1000); print('OK', X.shape, 'features:', d)\""
