from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, RESULTS_DIR
from src.interpret.attribution import proxy_attribution_from_weights
from src.interpret.info_efficiency import hypothesis_tests, summarize_modality_importance


def main(version: str = "v1.0.0") -> None:
    result_root = RESULTS_DIR / version
    rows = []
    for h in [1, 3]:
        pattern = f"**/M12__S7__H{h}__W14__E1*"
        for pred_path in result_root.glob(f"*M12__S7__H{h}__W14__E1*/predictions.parquet"):
            att_path = pred_path.parent / "attention.parquet"
            pred = pd.read_parquet(pred_path)
            att = pd.read_parquet(att_path) if att_path.exists() else pd.DataFrame({"date": pred["date"], "alpha_t": 0.5})
            merged = pred.merge(att, on="date", how="left")
            merged["horizon"] = h
            rows.append(proxy_attribution_from_weights(merged, alpha_col="alpha_t"))

    if not rows:
        print("[analyze_info_efficiency] no predictions found")
        return

    all_attr = pd.concat(rows, ignore_index=True)
    out_dir = ARTIFACTS_DIR / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_attr.to_parquet(out_dir / "F9_attribution_raw.parquet", index=False)
    summary = summarize_modality_importance(all_attr)
    summary.to_csv(out_dir / "T9_info_efficiency_summary.csv", index=False, encoding="utf-8")
    tests = hypothesis_tests(all_attr)
    pd.DataFrame([tests]).to_csv(out_dir / "T9_hypothesis_tests.csv", index=False, encoding="utf-8")
    print("[analyze_info_efficiency] done")


if __name__ == "__main__":
    main()

