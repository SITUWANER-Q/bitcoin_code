from src.config import RAW_DIR
from src.ingestion.download_macro import TICKERS, download_macro


if __name__ == "__main__":
    out_dir = RAW_DIR / "macro"
    expected = [out_dir / f"{name}.parquet" for name in TICKERS.keys()]
    if all(p.exists() and p.stat().st_size > 0 for p in expected):
        print(f"[download_macro] skip: all {len(expected)} parquet files already exist in {out_dir}")
    else:
        paths = download_macro()
        for path in paths:
            print(path)
