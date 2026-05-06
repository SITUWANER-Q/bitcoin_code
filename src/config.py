from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FROZEN_DIR = DATA_DIR / "frozen"
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


DATE_COL = "date"
TARGET_END_DATE = "2026-04-30"


def ensure_dirs() -> None:
    for path in [
        RAW_DIR / "ohlcv",
        RAW_DIR / "onchain",
        RAW_DIR / "macro",
        RAW_DIR / "twitter",
        RAW_DIR / "reddit",
        RAW_DIR / "news",
        PROCESSED_DIR,
        FROZEN_DIR,
        CONFIGS_DIR,
        RESULTS_DIR,
        LOGS_DIR,
        ARTIFACTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

