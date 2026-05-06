from __future__ import annotations

from pathlib import Path

import yfinance as yf

from src.config import RAW_DIR


def download_ohlcv(
    ticker: str = "BTC-USD",
    start: str = "2017-08-01",
    end: str = "2026-05-01",
    interval: str = "1d",
) -> Path:
    out_dir = RAW_DIR / "ohlcv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False)
    df = df.rename_axis("date").reset_index()
    out_path = out_dir / "btc_usd_daily.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    path = download_ohlcv()
    print(f"[download_ohlcv] saved: {path}")

