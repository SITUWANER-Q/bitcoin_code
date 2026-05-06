from __future__ import annotations

import numpy as np
from dieboldmariano import dm_test


def dm_hln(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, h: int = 1) -> dict[str, float]:
    err_a = y_true - y_pred_a
    err_b = y_true - y_pred_b
    stat, pval = dm_test(err_a, err_b, h=h, one_sided=False, harvey_correction=True)
    return {"dm_stat": float(stat), "p_value": float(pval)}

