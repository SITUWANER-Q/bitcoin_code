from __future__ import annotations

import numpy as np
import pandas as pd


def compute_positions(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.where(prob > threshold, 1.0, -1.0)


def run_backtest(
    returns: np.ndarray,
    cls_prob: np.ndarray,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    rv20: np.ndarray | None = None,
    mean_rv20: float | None = None,
) -> dict[str, float]:
    pos = compute_positions(cls_prob)
    trade = np.abs(np.diff(np.concatenate([[0.0], pos])))
    fee = (fee_bps / 1e4) * np.ones_like(returns)
    if rv20 is not None and mean_rv20 is not None and mean_rv20 > 0:
        rv20 = np.asarray(rv20, dtype=float)
        slip = (slippage_bps / 1e4) * (1.0 + (rv20 / mean_rv20))
    else:
        slip = (slippage_bps / 1e4) * np.ones_like(returns)
    friction = fee + slip
    pnl = pos * returns - trade * friction

    cum = np.cumprod(1 + pnl)
    total_return = float(cum[-1] - 1) if len(cum) else 0.0
    ann_ret = float((1 + total_return) ** (365 / max(len(pnl), 1)) - 1)
    ann_vol = float(np.std(pnl) * np.sqrt(365))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    downside = np.std(np.minimum(pnl, 0)) * np.sqrt(365)
    sortino = ann_ret / downside if downside > 0 else 0.0
    max_dd = float(np.max((np.maximum.accumulate(cum) - cum) / np.clip(np.maximum.accumulate(cum), 1e-8, None))) if len(cum) else 0.0

    gross_pos = pnl[pnl > 0].sum()
    gross_neg = -pnl[pnl < 0].sum()
    pf = float(gross_pos / gross_neg) if gross_neg > 0 else float("inf")
    turnover = float(np.mean(trade))
    return {
        "total_return": total_return,
        "cagr": ann_ret,
        "volatility": ann_vol,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "profit_factor": pf,
        "turnover": turnover,
    }

