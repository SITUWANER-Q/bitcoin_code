from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FoldSlice:
    fold: int
    train_mask: pd.Series
    val_mask: pd.Series
    test_mask: pd.Series


def make_5fold_tss(df: pd.DataFrame) -> list[FoldSlice]:
    dates = pd.to_datetime(df["date"])
    folds_def = [
        ("2018-01-01", "2020-12-31", "2021-01-01", "2021-03-31", "2021-04-01", "2021-06-30"),
        ("2018-01-01", "2021-06-30", "2021-07-01", "2021-09-30", "2021-10-01", "2021-12-31"),
        ("2018-01-01", "2022-06-30", "2022-07-01", "2022-09-30", "2022-10-01", "2023-03-31"),
        ("2018-01-01", "2023-06-30", "2023-07-01", "2023-09-30", "2023-10-01", "2024-03-31"),
        ("2018-01-01", "2024-09-30", "2024-10-01", "2024-12-31", "2025-01-01", "2026-03-31"),
    ]
    out = []
    for i, (tr_s, tr_e, v_s, v_e, te_s, te_e) in enumerate(folds_def, start=1):
        train_mask = (dates >= tr_s) & (dates <= tr_e)
        val_mask = (dates >= v_s) & (dates <= v_e)
        test_mask = (dates >= te_s) & (dates <= te_e)
        out.append(FoldSlice(i, train_mask, val_mask, test_mask))
    return out

