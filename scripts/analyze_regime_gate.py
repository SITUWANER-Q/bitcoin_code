from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, RESULTS_DIR
from src.interpret.attention_viz import plot_alpha_series
from src.interpret.regime_gate_events import event_window_alpha, pre_post_shift


def main(version: str = "v1.0.0") -> None:
    result_root = RESULTS_DIR / version
    event_path = Path("data/events.csv")
    events = pd.read_csv(event_path, encoding="utf-8")

    attention_parts = []
    for path in result_root.glob("main__M12__S7__H1__W14__E1__*/attention.parquet"):
        df = pd.read_parquet(path)
        attention_parts.append(df)
    if not attention_parts:
        print("[analyze_regime_gate] no attention files found")
        return
    att = pd.concat(attention_parts, ignore_index=True).groupby("date", as_index=False)["alpha_t"].mean()

    out_dir = ARTIFACTS_DIR / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_alpha_series(att, out_dir / "F5a_alpha_series.png")

    ew = event_window_alpha(att, events, window=10)
    ew.to_parquet(out_dir / "T8_event_window_alpha.parquet", index=False)
    shift = pre_post_shift(ew)
    shift.to_csv(out_dir / "T8_event_shift.csv", index=False, encoding="utf-8")
    print("[analyze_regime_gate] done")


if __name__ == "__main__":
    main()

