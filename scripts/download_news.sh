#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_NEWS="${ROOT}/data/raw/news"
mkdir -p "${RAW_NEWS}"

# Kaggle crypto-news drop lives directly in data/raw/news; gdelt has its own subdir.
existing_kaggle="$(find "${RAW_NEWS}" -maxdepth 2 -type f \( -name '*.csv' -o -name '*.parquet' \) -not -path '*/gdelt/*' -print -quit 2>/dev/null || true)"
if [ -n "${existing_kaggle}" ]; then
  echo "[download_news] skip kaggle: already have ${existing_kaggle}"
else
  kaggle datasets download -d oliviervha/crypto-news -p "${RAW_NEWS}" --unzip
fi

# GDELT: per-day skip is implemented inside src/ingestion/download_gdelt.py.
python "${ROOT}/scripts/download_gdelt.py"

echo "[download_news] done"
