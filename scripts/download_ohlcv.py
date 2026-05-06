from src.config import RAW_DIR
from src.ingestion.download_ohlcv import download_ohlcv


if __name__ == "__main__":
    out_path = RAW_DIR / "ohlcv" / "btc_usd_daily.parquet"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[download_ohlcv] skip: already exists at {out_path}")
    else:
        path = download_ohlcv()
        print(path)
