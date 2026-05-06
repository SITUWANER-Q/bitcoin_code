#!/usr/bin/env bash
set -euo pipefail

python scripts/build_reports.py
python scripts/analyze_regime_gate.py
python scripts/analyze_info_efficiency.py
python scripts/analyze_break_even.py

echo "[generate_paper_assets] artifacts ready in artifacts/"
