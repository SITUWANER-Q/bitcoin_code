from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.config import TARGET_END_DATE


@dataclass
class AlignedData:
    full: pd.DataFrame
    numerical_cols: list[str]
    text_cols: list[str]
    label_cols: list[str]


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out


def _build_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_logret_t1"] = out["log_return"].shift(-1)
    out["target_logret_t3"] = np.log(out["close"].shift(-3) / out["close"])
    out["target_dir_t1"] = (out["target_logret_t1"] > 0).astype(float)
    out["target_dir_t3"] = (out["target_logret_t3"] > 0).astype(float)
    out["target_vol_gk_t1"] = out["vol_gk_5d"].shift(-1)
    out["target_vol_gk_t3"] = out["vol_gk_20d"].shift(-3)
    return out


def align_modalities(
    numerical_df: pd.DataFrame,
    onchain_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> AlignedData:
    num = _ensure_date(numerical_df)
    on = _ensure_date(onchain_df)
    ma = _ensure_date(macro_df)
    se = _ensure_date(sentiment_df)
    text_cols_raw = [c for c in se.columns if c.startswith("score_") or c.startswith("emb_") or c.startswith("count_")]
    se_original = se.copy()
    if text_cols_raw:
        # Strict anti-leak: t-day prediction can only use textual aggregates available before t 00:00 UTC.
        se = se.copy()
        se["date"] = se["date"] + pd.Timedelta(days=1)

    merged = num.merge(on, on="date", how="left").merge(ma, on="date", how="left").merge(se, on="date", how="left")
    merged = merged.sort_values("date")
    merged = merged[merged["date"] <= pd.to_datetime(TARGET_END_DATE)].copy()
    merged = _build_labels(merged)

    text_cols = [c for c in merged.columns if c.startswith("score_") or c.startswith("emb_") or c.startswith("count_")]
    numeric_cols = [
        c
        for c in merged.columns
        if c
        not in {"date", *text_cols, "target_logret_t1", "target_logret_t3", "target_dir_t1", "target_dir_t3", "target_vol_gk_t1", "target_vol_gk_t3"}
    ]

    merged[text_cols] = merged[text_cols].ffill(limit=3)
    merged[numeric_cols] = merged[numeric_cols].ffill(limit=1)

    robust = RobustScaler()
    minmax = MinMaxScaler()
    merged[numeric_cols] = robust.fit_transform(merged[numeric_cols])
    merged[numeric_cols] = minmax.fit_transform(merged[numeric_cols])

    label_cols = [
        "target_logret_t1",
        "target_logret_t3",
        "target_dir_t1",
        "target_dir_t3",
        "target_vol_gk_t1",
        "target_vol_gk_t3",
    ]

    merged = merged.dropna(subset=label_cols).reset_index(drop=True)
    if text_cols_raw and len(merged) > 1:
        # Explicit cross-check: after lagging, first row should not mirror same-day raw textual feature.
        probe_col = "score_twitter" if "score_twitter" in merged.columns else text_cols_raw[0]
        original_by_date = se_original.set_index("date")
        first_date = merged.iloc[0]["date"]
        first_lagged_val = merged.iloc[0][probe_col]
        if first_date in original_by_date.index and probe_col in original_by_date.columns:
            first_raw_val = original_by_date.loc[first_date, probe_col]
            if isinstance(first_raw_val, pd.Series):
                first_raw_val = first_raw_val.iloc[0]
            if pd.notna(first_lagged_val) and pd.notna(first_raw_val):
                assert not np.isclose(
                    float(first_lagged_val), float(first_raw_val), rtol=0.0, atol=0.0
                ), "Check lagging logic! first-row textual feature still matches same-day raw signal."
    assert (merged.groupby("date")["target_logret_t1"].count() <= 1).all(), "duplicate date rows detected after alignment"
    return AlignedData(full=merged, numerical_cols=numeric_cols, text_cols=text_cols, label_cols=label_cols)

