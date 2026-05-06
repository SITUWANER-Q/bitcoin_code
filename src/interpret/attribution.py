from __future__ import annotations

import numpy as np
import pandas as pd


def proxy_attribution_from_weights(
    pred_df: pd.DataFrame,
    alpha_col: str = "alpha_t",
) -> pd.DataFrame:
    """
    Lightweight attribution proxy.
    If full Captum run is not requested, derive modality proportions from alpha.
    """
    df = pred_df.copy()
    alpha = df[alpha_col].fillna(0.5).values
    out = pd.DataFrame(
        {
            "date": df["date"],
            "horizon": df.get("horizon", 1),
            "attr_price": 0.35 + 0.15 * (1 - alpha),
            "attr_tech": 0.20 + 0.10 * (1 - alpha),
            "attr_onchain": 0.15 + 0.10 * (1 - alpha),
            "attr_twitter": 0.20 + 0.20 * alpha,
            "attr_news_reddit": 0.10 + 0.25 * alpha,
        }
    )
    cols = ["attr_price", "attr_tech", "attr_onchain", "attr_twitter", "attr_news_reddit"]
    s = out[cols].sum(axis=1).replace(0, np.nan)
    out[cols] = out[cols].div(s, axis=0)
    return out

