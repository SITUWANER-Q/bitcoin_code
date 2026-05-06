#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/news
kaggle datasets download -d oliviervha/crypto-news -p data/raw/news --unzip
python scripts/download_gdelt.py

echo "[download_news] done"
