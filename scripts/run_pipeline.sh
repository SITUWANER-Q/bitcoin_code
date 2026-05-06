#!/usr/bin/env bash
set -euo pipefail

bash scripts/download_all.sh
python scripts/freeze_data.py --version v1.0.0 --encoders E1 E2 E3
python scripts/verify_data.py --version v1.0.0
python scripts/generate_configs.py
python scripts/run_batch.py --smoke
python scripts/run_batch.py main
