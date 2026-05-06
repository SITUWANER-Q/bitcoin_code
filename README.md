# BTC Multimodal Forecasting

Production-ready research codebase for top-tier multimodal BTC forecasting experiments.

## Project Layout

- `src/ingestion`: download and parse raw data.
- `src/features`: numerical/text feature engineering and alignment.
- `src/models`: M1-M12 models and A1-A8 ablations.
- `src/training`: seed control, folds, training loops.
- `src/eval`: metrics, backtest, statistical tests.
- `src/interpret`: attention and attribution analyses.
- `src/runners`: experiment entrypoints.
- `scripts`: orchestration scripts for full pipeline.

# 克隆（二选一）
git clone https://github.com/SITUWANER-Q/bitcoin_code 
cd bitcoin_code 


```bash
bash setup.sh btc 3.10

conda activate btc
bash scripts/download_all.sh
python scripts/freeze_data.py
python scripts/generate_configs.py
python scripts/run_batch.py --smoke
python scripts/run_batch.py main
```