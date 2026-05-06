#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/reddit
kaggle datasets download -d pavellexyr/the-reddit-crypto-dataset -p data/raw/reddit --unzip

echo "[download_reddit] done"
