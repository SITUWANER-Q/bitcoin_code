#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/{ohlcv,onchain,macro,twitter,reddit,news} data/processed data/frozen

python scripts/download_ohlcv.py
bash scripts/download_onchain.sh
python scripts/download_macro.py
bash scripts/download_twitter.sh
bash scripts/download_reddit.sh
bash scripts/download_news.sh

echo "[download_all] all sources downloaded"
