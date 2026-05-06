#!/usr/bin/env bash
# Drop -e so a failure in one source does not abort the rest.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
mkdir -p data/raw/{ohlcv,onchain,macro,twitter,reddit,news} data/processed data/frozen

run_step() {
  local name="$1"; shift
  echo "[download_all] >> ${name}"
  if "$@"; then
    echo "[download_all] ${name}: ok"
  else
    echo "[download_all] ${name}: FAILED (continuing)"
  fi
}

run_step ohlcv   python scripts/download_ohlcv.py
run_step onchain bash   scripts/download_onchain.sh
run_step macro   python scripts/download_macro.py
run_step twitter bash   scripts/download_twitter.sh
run_step reddit  bash   scripts/download_reddit.sh
run_step news    bash   scripts/download_news.sh

echo "[download_all] all sources attempted"
