from __future__ import annotations

import numpy as np
import pandas as pd


def simple_mcs(loss_df: pd.DataFrame, alpha: float = 0.10) -> pd.DataFrame:
    """
    Lightweight approximation of Model Confidence Set.
    Keeps models whose mean loss is within (1+alpha) of best.
    """
    means = loss_df.mean(axis=0).sort_values()
    best = means.iloc[0]
    cutoff = best * (1 + alpha)
    out = pd.DataFrame({"model": means.index, "mean_loss": means.values})
    out["in_mcs"] = out["mean_loss"] <= cutoff
    return out

