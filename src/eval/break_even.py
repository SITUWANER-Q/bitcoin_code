from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.backtest import run_backtest


def cost_grid_scan(
    returns: np.ndarray,
    cls_prob: np.ndarray,
    model_name: str,
    cost_grid_bps: list[float] | None = None,
    rv20: np.ndarray | None = None,
    mean_rv20: float | None = None,
    dynamic_slippage: bool = True,
) -> pd.DataFrame:
    grid = cost_grid_bps or [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    rows = []
    for c in grid:
        stats = run_backtest(
            returns,
            cls_prob,
            fee_bps=c,
            slippage_bps=c / 2,
            rv20=rv20 if dynamic_slippage else None,
            mean_rv20=mean_rv20 if dynamic_slippage else None,
        )
        rows.append({"model": model_name, "cost_bps": c, **stats})
    return pd.DataFrame(rows)


def break_even_from_curve(curve: pd.DataFrame, sharpe_floor: float = 1.0) -> dict[str, float]:
    curve = curve.sort_values("cost_bps")
    bef_sharpe = float("nan")
    for _, row in curve.iterrows():
        if row["sharpe"] <= sharpe_floor:
            bef_sharpe = float(row["cost_bps"])
            break
    bef_pf1 = float("nan")
    for _, row in curve.iterrows():
        if row["profit_factor"] <= 1.0:
            bef_pf1 = float(row["cost_bps"])
            break
    return {"bef_sharpe": bef_sharpe, "bef_pf1": bef_pf1}

