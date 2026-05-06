# BTC Multimodal Forecasting

Production-ready research codebase for top-tier multimodal BTC forecasting experiments.

## Quick Start (Linux + RTX 4090)

```bash
bash setup.sh btc 3.10
bash scripts/download_all.sh
python scripts/freeze_data.py
python scripts/generate_configs.py
python scripts/run_batch.py --smoke
python scripts/run_batch.py main
```

## Project Layout

- `src/ingestion`: download and parse raw data.
- `src/features`: numerical/text feature engineering and alignment.
- `src/models`: M1-M12 models and A1-A8 ablations.
- `src/training`: seed control, folds, training loops.
- `src/eval`: metrics, backtest, statistical tests.
- `src/interpret`: attention and attribution analyses.
- `src/runners`: experiment entrypoints.
- `scripts`: orchestration scripts for full pipeline.
