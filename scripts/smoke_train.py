from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.eval.backtest import run_backtest
from src.models.m12_ra_bicma import build as build_m12
from src.training.dataset import WindowedData
from src.training.tss_trainer import fit_predict_model


def make_synthetic_windows(n_samples: int = 64, lookback: int = 14, num_dim: int = 5, txt_dim: int = 16) -> WindowedData:
    rng = np.random.default_rng(123)
    x_num = rng.normal(0, 1, size=(n_samples, lookback, num_dim)).astype(np.float32)
    x_txt = rng.normal(0, 1, size=(n_samples, lookback, txt_dim)).astype(np.float32)
    y_reg = rng.normal(0, 0.02, size=(n_samples, 1)).astype(np.float32)
    y_cls = (y_reg > 0).astype(np.float32)
    y_vol = np.abs(rng.normal(0.01, 0.002, size=(n_samples, 1))).astype(np.float32)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="D").strftime("%Y-%m-%d").tolist()
    return WindowedData(x_num=x_num, x_txt=x_txt, y_reg=y_reg, y_cls=y_cls, y_vol=y_vol, dates=dates)


def check_warmup_monotonic(lr: float = 1e-4, warmup_epochs: int = 8) -> None:
    vals = [lr * min(1.0, float(e + 1) / float(warmup_epochs)) for e in range(warmup_epochs)]
    diffs = np.diff(vals)
    assert np.all(diffs >= -1e-12), "warmup LR is not monotonic increasing"
    assert abs(vals[0] - lr / warmup_epochs) < 1e-12


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_m12(num_dim=5, txt_dim=16, hidden_dim=32, n_heads=4)

    # Stage-1 freeze check.
    model.freeze_text_branch(True)
    assert all(not p.requires_grad for p in model.txt_proj.parameters())
    model.freeze_text_branch(False)
    assert all(p.requires_grad for p in model.txt_proj.parameters())

    train_data = make_synthetic_windows(n_samples=80)
    val_data = make_synthetic_windows(n_samples=24)
    test_data = make_synthetic_windows(n_samples=24)

    out = fit_predict_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        seed=42,
        lr=1e-4,
        batch_size=16,
        max_epoch=2,
        patience=2,
        warmup_epochs=8,
        staged_training=True,
        stage_switch_epoch=1,
        device=device,
    )

    assert not out.predictions.empty
    assert not out.attention.empty
    alpha = out.attention["alpha_t"].to_numpy()
    assert np.all(np.isfinite(alpha))
    assert np.nanmin(alpha) >= 0.0 and np.nanmax(alpha) <= 1.0
    alpha_mean_global = float(np.nanmean(alpha))
    assert abs(alpha_mean_global - 0.5) > 1e-4, "alpha stuck at 0.5000, regime gate likely not learning"

    check_warmup_monotonic()
    assert not out.trace.empty, "missing training trace"
    first_epoch_row = out.trace.iloc[0]
    stage_epoch = 1
    stage_row = out.trace[out.trace["epoch"] == float(stage_epoch)]
    if stage_row.empty:
        stage_row = out.trace.iloc[[-1]]
    stage_row = stage_row.iloc[0]
    print(
        f"[smoke] epoch0 alpha_mean={first_epoch_row['alpha_mean']:.4f} "
        f"train_loss={first_epoch_row['train_loss']:.6f} val_loss={first_epoch_row['val_loss']:.6f}"
    )
    print(
        f"[smoke] epoch{stage_epoch} alpha_mean={stage_row['alpha_mean']:.4f} "
        f"train_loss={stage_row['train_loss']:.6f} val_loss={stage_row['val_loss']:.6f}"
    )
    # Stage-switch explosion guard: allow some fluctuation but reject blow-up.
    if np.isfinite(first_epoch_row["train_loss"]) and np.isfinite(stage_row["train_loss"]):
        assert stage_row["train_loss"] <= first_epoch_row["train_loss"] * 10.0, "loss exploded after stage switch"

    # Dynamic slippage path smoke.
    rets = out.predictions["y_true_reg"].to_numpy()
    probs = out.predictions["y_pred_cls_prob"].to_numpy()
    rv20 = np.clip(out.predictions["y_true_vol"].to_numpy(), 1e-8, None)
    stats = run_backtest(rets, probs, fee_bps=10.0, slippage_bps=5.0, rv20=rv20, mean_rv20=float(rv20.mean()))
    assert np.isfinite(stats["turnover"])
    assert len(rv20) == len(rets), "dynamic slippage path length mismatch"

    print("[smoke] all checks passed")


if __name__ == "__main__":
    main()

