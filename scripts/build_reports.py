from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, RESULTS_DIR
from src.eval.significance import collect_predictions, dm_matrix, mcs_from_predictions


def main(version: str = "v1.0.0") -> None:
    result_root = RESULTS_DIR / version
    out_dir = ARTIFACTS_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for path in result_root.glob("*/metrics.parquet"):
        m = pd.read_parquet(path)
        m["config_id"] = path.parent.name
        metric_rows.append(m)
    if metric_rows:
        all_metrics = pd.concat(metric_rows, ignore_index=True)
        all_metrics.to_parquet(out_dir / "all_metrics.parquet", index=False)
        # table-style rollups
        table_main = all_metrics[all_metrics["config_id"].str.startswith("main__")]
        table_main.to_csv(out_dir / "table_main.csv", index=False, encoding="utf-8")

    preds = collect_predictions(result_root)
    if not preds.empty:
        preds.to_parquet(out_dir / "all_predictions.parquet", index=False)
        dm = dm_matrix(preds)
        dm.to_csv(out_dir / "T7_dm_matrix.csv", index=False, encoding="utf-8")
        mcs = mcs_from_predictions(preds)
        mcs.to_csv(out_dir / "T7_mcs.csv", index=False, encoding="utf-8")
    print("[build_reports] done")


if __name__ == "__main__":
    main()

