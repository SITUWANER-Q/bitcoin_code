#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/twitter
kaggle datasets download -d kaushiksuresh147/bitcoin-tweets -p data/raw/twitter --unzip
# Removed: gauravduttakiit/bitcoin-tweets (no longer available on Kaggle / API 403).
# Optional extra Twitter data: place CSV/Parquet under data/raw/twitter (e.g. HF
# StephanAkkerman/bitcoin-tweets-2021-2025 or Kaggle varpit94/bitcoin-tweets-2022-2026).

echo "[download_twitter] done"
