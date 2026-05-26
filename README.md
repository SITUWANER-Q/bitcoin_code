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

## Quick Start

```bash
git clone https://github.com/SITUWANER-Q/bitcoin_code.git
cd bitcoin_code
bash setup.sh btc 3.10
conda activate btc
bash scripts/download_all.sh  
python scripts/freeze_data.py
python scripts/generate_configs.py
python scripts/run_batch.py --smoke
python scripts/run_batch.py main
```

`setup.sh` 已含 `pip install -e .`。大数据放数据盘（如 AutoDL：`/root/autodl-tmp`）。

## Git

```bash
git pull && git add . && git commit -m "update" 
git push
```

---

## 服务器与环境（简）

1. **外网（AutoDL）**：新 SSH 按需执行 `source /etc/network_turbo`。
2. **克隆到数据盘**：`cd /root/autodl-tmp && git clone … && cd bitcoin_code`。
3. **pip 加速**：官方 PyPI 很慢时先 `export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` 再跑 `bash setup.sh btc 3.10`。若出现 `from versions: none`，多为坏镜像（避免仅 `http://mirrors.aliyun.com`）；可改清华或 `pip install -i https://pypi.org/simple -r requirements.txt`。
4. **PyTorch**：`setup.sh` 会装 `requirements-cu121.txt`；仍慢可再单独用官方索引装 torch。
5. **Kaggle**：`~/.kaggle/kaggle.json`，`chmod 600`。
6. **Yahoo 429**：隔段时间重跑 `python scripts/download_ohlcv.py`、`python scripts/download_macro.py`，或拷齐 `data/raw/ohlcv`、`data/raw/macro`。
7. **断线续跑**：`tmux new -s train`，会话里 `conda activate` + `run_batch`；恢复用 `tmux attach -t train`。

自检：`nvidia-smi`，`python -c "import torch,src; print(torch.cuda.is_available())"`。`main` 约 720 个配置；多卡：`python scripts/run_batch.py main --n-gpus 2`。

## 维护记录

- **2026-05-07** — pip 镜像无版本或极慢：`PIP_INDEX_URL` 清华，或 `-i https://pypi.org/simple`，配合 `network_turbo`。
- **2026-05-07** — Yahoo 429：重试或同步 raw。
- **2026-05-07** — dieboldmariano：`requirements.txt` 已锁定 `1.1.0`。
- **2026-05-25** — GDELT：关键词改为 `(bitcoin OR btc OR cryptocurrency)`；长期下载用 `bash scripts/gdelt_loop.sh`（带单实例锁）。
