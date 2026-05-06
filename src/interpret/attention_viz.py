from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_alpha_series(attention_df: pd.DataFrame, out_path: Path) -> None:
    tmp = attention_df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp = tmp.sort_values("date")
    plt.figure(figsize=(14, 4))
    plt.plot(tmp["date"], tmp["alpha_t"], linewidth=1.2)
    plt.ylim(0, 1)
    plt.title("Regime Gate Alpha Over Time")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

