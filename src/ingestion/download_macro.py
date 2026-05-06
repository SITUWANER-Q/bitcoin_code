from __future__ import annotations

from pathlib import Path

import yfinance as yf

from src.config import RAW_DIR


TICKERS = {
    "dxy": "DX-Y.NYB",
    "spx": "^GSPC",
    "vix": "^VIX",
    "gold": "GC=F",
    "us10y": "^TNX",
    "nasdaq": "^IXIC",
}


def download_macro(start: str = "2017-08-01", end: str = "2026-05-01") -> list[Path]:
    out_dir = RAW_DIR / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, ticker in TICKERS.items():
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False)
        df = df.rename_axis("date").reset_index()
        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        paths.append(out_path)
    return paths


if __name__ == "__main__":
    for p in download_macro():
        print(f"[download_macro] saved: {p}")

