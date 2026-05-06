from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.features.aligner import AlignedData, align_modalities
from src.features.sentiment_encoder import SentimentEncoder, aggregate_daily_sentiment
from src.features.ta_indicators import build_numerical_features
from src.ingestion.loaders import (
    load_macro,
    load_news_raw,
    load_ohlcv,
    load_onchain,
    load_reddit_raw,
    load_twitter_raw,
)


def _macro_to_returns(macro_df: pd.DataFrame) -> pd.DataFrame:
    out = macro_df.copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in [c for c in out.columns if c.endswith("_close")]:
        out[col] = (out[col].astype(float)).pct_change()
    return out


def build_sentiment_table(
    encoders: list[str],
    max_docs_per_source: int | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    sources = {
        "twitter": load_twitter_raw(),
        "reddit": load_reddit_raw(),
        "news": load_news_raw(),
    }
    all_tables = []
    for encoder_id in encoders:
        encoder = SentimentEncoder(encoder_id=encoder_id)
        source_tables = []
        for source_name, df in sources.items():
            if df.empty:
                continue
            data = df.copy()
            if max_docs_per_source is not None and len(data) > max_docs_per_source:
                data = data.sample(n=max_docs_per_source, random_state=42)
            data["text"] = data["text"].astype(str).fillna("")
            table = aggregate_daily_sentiment(
                data[["date", "text"]],
                encoder,
                f"{source_name}_{encoder_id.lower()}",
                n_jobs=n_jobs,
            )
            source_tables.append(table)
        if not source_tables:
            continue
        merged = source_tables[0]
        for t in source_tables[1:]:
            merged = merged.merge(t, on="date", how="outer")
        all_tables.append(merged)
    if not all_tables:
        return pd.DataFrame(columns=["date"])
    out = all_tables[0]
    for t in all_tables[1:]:
        out = out.merge(t, on="date", how="outer")
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date")


def build_aligned_dataset(
    encoders: list[str] | None = None,
    max_docs_per_source: int | None = None,
    n_jobs: int = -1,
) -> AlignedData:
    ohlcv = load_ohlcv()
    onchain = load_onchain()
    macro = _macro_to_returns(load_macro())
    numerical = build_numerical_features(ohlcv)
    sentiment = build_sentiment_table(
        encoders=encoders or ["E1"],
        max_docs_per_source=max_docs_per_source,
        n_jobs=n_jobs,
    )
    return align_modalities(numerical, onchain, macro, sentiment)

