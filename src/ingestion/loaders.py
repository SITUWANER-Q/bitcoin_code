from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR


def load_ohlcv() -> pd.DataFrame:
    path = RAW_DIR / "ohlcv" / "btc_usd_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    return df[["date", "open", "high", "low", "close", "volume"]]


def load_onchain() -> pd.DataFrame:
    path = RAW_DIR / "onchain" / "btc.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8")
    cols = [
        "time",
        "AdrActCnt",
        "AdrBalUSD1Cnt",
        "AdrBalUSD1KCnt",
        "BlkSizeMeanByte",
        "CapMrktCurUSD",
        "CapRealUSD",
        "FeeMeanUSD",
        "FeeTotUSD",
        "HashRate",
        "IssContPctAnn",
        "NVTAdj",
        "NVTAdj90",
        "SplyCur",
        "TxCnt",
        "TxTfrValAdjUSD",
        "TxTfrValMeanUSD",
        "VtyDayRet180d",
        "VtyDayRet30d",
    ]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].rename(columns={"time": "date"}).copy()
    out["date"] = pd.to_datetime(out["date"])
    if "CapMrktCurUSD" in out.columns and "CapRealUSD" in out.columns:
        out["mvrv"] = out["CapMrktCurUSD"] / out["CapRealUSD"].replace(0, pd.NA)
    return out


def load_macro() -> pd.DataFrame:
    macro_dir = RAW_DIR / "macro"
    parts = []
    for path in sorted(macro_dir.glob("*.parquet")):
        name = path.stem
        raw = pd.read_parquet(path)
        df = raw.rename(columns={c: c.lower() for c in raw.columns})
        if "adj close" in df.columns:
            price_col = "adj close"
        elif "close" in df.columns:
            price_col = "close"
        else:
            continue
        tmp = df[["date", price_col]].rename(columns={price_col: f"{name}_close"})
        parts.append(tmp)
    if not parts:
        return pd.DataFrame(columns=["date"])
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="date", how="outer")
    return out.sort_values("date")


def _read_kaggle_csvs(folder: Path) -> pd.DataFrame:
    frames = []
    for csv_path in folder.glob("**/*.csv"):
        try:
            frames.append(pd.read_csv(csv_path, encoding="utf-8", low_memory=False))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_twitter_raw() -> pd.DataFrame:
    df = _read_kaggle_csvs(RAW_DIR / "twitter")
    if df.empty:
        return df
    date_col = "date" if "date" in df.columns else "created_at"
    text_col = "text" if "text" in df.columns else "tweet"
    followers_col = "user_followers" if "user_followers" in df.columns else None
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "text": df[text_col].astype(str),
            "user_followers": pd.to_numeric(df[followers_col], errors="coerce") if followers_col else 0.0,
        }
    )
    return out.dropna(subset=["date"]).reset_index(drop=True)


def load_reddit_raw() -> pd.DataFrame:
    df = _read_kaggle_csvs(RAW_DIR / "reddit")
    if df.empty:
        return df
    date_col = "created_utc" if "created_utc" in df.columns else "date"
    text_col = "body" if "body" in df.columns else ("text" if "text" in df.columns else "title")
    subreddit_col = "subreddit" if "subreddit" in df.columns else None
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce", unit="s" if date_col == "created_utc" else None),
            "text": df[text_col].astype(str),
            "subreddit": df[subreddit_col].astype(str) if subreddit_col else "unknown",
        }
    )
    out = out.dropna(subset=["date"])
    if "subreddit" in out.columns:
        out = out[out["subreddit"].str.lower().isin({"bitcoin", "cryptocurrency", "btc"})]
    return out.reset_index(drop=True)


def load_news_raw() -> pd.DataFrame:
    kaggle = _read_kaggle_csvs(RAW_DIR / "news")
    records = []
    gdelt_dir = RAW_DIR / "news" / "gdelt"
    for path in gdelt_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for art in payload.get("articles", []):
                records.append(
                    {
                        "date": pd.to_datetime(art.get("seendate"), errors="coerce"),
                        "text": art.get("title", ""),
                    }
                )
        except Exception:
            continue
    gdelt = pd.DataFrame(records)
    if not kaggle.empty:
        dcol = "date" if "date" in kaggle.columns else kaggle.columns[0]
        tcol = "title" if "title" in kaggle.columns else ("text" if "text" in kaggle.columns else kaggle.columns[-1])
        kaggle = pd.DataFrame({"date": pd.to_datetime(kaggle[dcol], errors="coerce"), "text": kaggle[tcol].astype(str)})
    out = pd.concat([kaggle, gdelt], ignore_index=True) if not kaggle.empty or not gdelt.empty else pd.DataFrame(columns=["date", "text"])
    return out.dropna(subset=["date"]).reset_index(drop=True)

