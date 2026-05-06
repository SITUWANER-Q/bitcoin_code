#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_TWITTER="${ROOT}/data/raw/twitter"
mkdir -p "${RAW_TWITTER}"

# Match loaders.py: twitter raw is any recursive *.csv or *.parquet under this folder.
existing="$(find "${RAW_TWITTER}" -type f \( -name '*.csv' -o -name '*.parquet' \) -print -quit 2>/dev/null || true)"
if [ -n "${existing}" ]; then
  echo "[download_twitter] skip: already have twitter raw at ${existing}"
  exit 0
fi

kaggle datasets download -d kaushiksuresh147/bitcoin-tweets -p "${RAW_TWITTER}" --unzip

echo "[download_twitter] done"
