#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/data/raw/onchain/btc.csv"
mkdir -p "$(dirname "${OUT}")"

if [ -s "${OUT}" ]; then
  echo "[download_onchain] skip: already have ${OUT}"
  exit 0
fi

curl -L -o "${OUT}" \
  "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"

echo "[download_onchain] done"
