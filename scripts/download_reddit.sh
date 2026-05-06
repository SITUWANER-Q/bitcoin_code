#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_REDDIT="${ROOT}/data/raw/reddit"
mkdir -p "${RAW_REDDIT}"

# Match loaders.py: reddit raw is any recursive *.csv or *.parquet under this folder.
existing="$(find "${RAW_REDDIT}" -type f \( -name '*.csv' -o -name '*.parquet' \) -print -quit 2>/dev/null || true)"
if [ -n "${existing}" ]; then
  echo "[download_reddit] skip: already have reddit raw at ${existing}"
  exit 0
fi

# Original source pavellexyr/the-reddit-crypto-dataset is no longer available on Kaggle.
# Replacement: gpreda/reddit-cryptocurrency (r/CryptoCurrency posts, actively maintained).
kaggle datasets download -d gpreda/reddit-cryptocurrency -p "${RAW_REDDIT}" --unzip

echo "[download_reddit] done"
