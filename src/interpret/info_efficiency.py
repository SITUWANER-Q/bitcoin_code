from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_modality_importance(attribution_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns:
    horizon, regime, attr_price, attr_tech, attr_onchain, attr_twitter, attr_news_reddit
    """
    cols = ["attr_price", "attr_tech", "attr_onchain", "attr_twitter", "attr_news_reddit"]
    out = attribution_df.groupby(["horizon"])[cols].agg(["mean", "std"])
    out.columns = ["_".join(c) for c in out.columns]
    out = out.reset_index()
    return out


def hypothesis_tests(attribution_df: pd.DataFrame) -> dict[str, float]:
    t1 = attribution_df[attribution_df["horizon"] == 1]["attr_twitter"].dropna().values
    t3 = attribution_df[attribution_df["horizon"] == 3]["attr_twitter"].dropna().values
    o1 = attribution_df[attribution_df["horizon"] == 1]["attr_onchain"].dropna().values
    o3 = attribution_df[attribution_df["horizon"] == 3]["attr_onchain"].dropna().values
    # simple z-style statistic proxy, avoids dependency on heavy stats package
    h1 = float(np.mean(t1) - np.mean(t3)) if len(t1) and len(t3) else float("nan")
    h2 = float(np.mean(o3) - np.mean(o1)) if len(o1) and len(o3) else float("nan")
    return {"h1_twitter_t1_gt_t3_effect": h1, "h2_onchain_t3_gt_t1_effect": h2}

