from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.dm_test import dm_hln
from src.eval.mcs import simple_mcs


def collect_predictions(result_root: Path) -> pd.DataFrame:
    rows = []
    for pred_path in result_root.glob("*/predictions.parquet"):
        config_id = pred_path.parent.name
        df = pd.read_parquet(pred_path)
        df["config_id"] = config_id
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def dm_matrix(pred_df: pd.DataFrame, anchor_prefix: str = "main__M12__S7__H1__W14__E1") -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    anchor = pred_df[pred_df["config_id"].str.startswith(anchor_prefix)]
    # collapse anchor by date average
    anchor_mean = anchor.groupby("date")[["y_true_reg", "y_pred_reg"]].mean().reset_index()
    rows = []
    for cid, g in pred_df.groupby("config_id"):
        merged = anchor_mean.merge(g[["date", "y_true_reg", "y_pred_reg"]], on="date", suffixes=("_a", "_b"))
        if merged.empty:
            continue
        dm = dm_hln(
            y_true=merged["y_true_reg_a"].values,
            y_pred_a=merged["y_pred_reg_a"].values,
            y_pred_b=merged["y_pred_reg_b"].values,
            h=1,
        )
        rows.append({"config_id": cid, **dm})
    return pd.DataFrame(rows).sort_values("p_value")


def mcs_from_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    pred_df = pred_df.copy()
    pred_df["sq_err"] = (pred_df["y_true_reg"] - pred_df["y_pred_reg"]) ** 2
    pivot = pred_df.pivot_table(index="date", columns="config_id", values="sq_err", aggfunc="mean")
    return simple_mcs(pivot.fillna(pivot.mean()))

