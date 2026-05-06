from __future__ import annotations

import re

import pandas as pd

from src.config import ARTIFACTS_DIR, RESULTS_DIR
from src.eval.break_even import break_even_from_curve, cost_grid_scan


def model_from_config_id(config_id: str) -> str:
    parts = config_id.split("__")
    for p in parts:
        if re.match(r"^(M\d+|A\d+)$", p):
            return p
    return "UNK"


def main(version: str = "v1.0.0") -> None:
    result_root = RESULTS_DIR / version
    curves = []
    summary = []
    for pred_path in result_root.glob("*/predictions.parquet"):
        cid = pred_path.parent.name
        pred = pd.read_parquet(pred_path)
        if pred.empty or "y_true_reg" not in pred.columns:
            continue
        model = model_from_config_id(cid)
        curve = cost_grid_scan(pred["y_true_reg"].values, pred["y_pred_cls_prob"].values, model_name=model)
        curve["config_id"] = cid
        curves.append(curve)
        be = break_even_from_curve(curve, sharpe_floor=1.0)
        summary.append({"config_id": cid, "model": model, **be, "turnover_year": curve["turnover"].mean() * 365})

    out_dir = ARTIFACTS_DIR / "cost"
    out_dir.mkdir(parents=True, exist_ok=True)
    if curves:
        all_curves = pd.concat(curves, ignore_index=True)
        all_curves.to_parquet(out_dir / "F10_break_even_curves.parquet", index=False)
    if summary:
        pd.DataFrame(summary).to_csv(out_dir / "T10_break_even_summary.csv", index=False, encoding="utf-8")
    print("[analyze_break_even] done")


if __name__ == "__main__":
    main()

