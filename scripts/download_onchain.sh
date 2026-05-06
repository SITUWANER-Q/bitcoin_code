#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/onchain
curl -L -o data/raw/onchain/btc.csv \
  "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"

echo "[download_onchain] done"
