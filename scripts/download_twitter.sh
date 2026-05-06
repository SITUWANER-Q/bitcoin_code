#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/twitter
kaggle datasets download -d kaushiksuresh147/bitcoin-tweets -p data/raw/twitter --unzip
kaggle datasets download -d gauravduttakiit/bitcoin-tweets -p data/raw/twitter --unzip

echo "[download_twitter] done"
