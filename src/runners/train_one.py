from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import FROZEN_DIR
from src.eval.backtest import run_backtest
from src.eval.metrics import classification_metrics, regression_metrics, volatility_metrics
from src.models.factory import build_neural_model, run_classic_model
from src.runners.config_schema import RunConfig
from src.training.folds import make_5fold_tss
from src.training.seed_utils import lock_seed
from src.training.tss_trainer import build_windows, fit_predict_model
from src.utils.io import read_json


def _load_frozen(version: str) -> pd.DataFrame:
    root = FROZEN_DIR / version
    _ = read_json(root / "MANIFEST.json")
    num = pd.read_parquet(root / "numerical_daily.parquet")
    txt = pd.read_parquet(root / "sentiment_daily.parquet")
    y = pd.read_parquet(root / "labels_daily.parquet")
    df = num.merge(txt, on="date", how="inner").merge(y, on="date", how="inner")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _scenario_columns(df: pd.DataFrame, scenario: str, encoder: str) -> tuple[list[str], list[str]]:
    num_cols = [c for c in df.columns if c not in {"date"} and not c.startswith("score_") and not c.startswith("emb_") and not c.startswith("count_") and not c.startswith("target_")]
    all_txt = [c for c in df.columns if c.startswith("score_") or c.startswith("emb_") or c.startswith("count_")]
    txt_encoder_cols = [c for c in all_txt if f"_{encoder.lower()}" in c]
    if scenario == "S1":
        return num_cols, []
    if scenario == "S4":
        txt = [c for c in txt_encoder_cols if c.startswith("score_") or c.startswith("emb_")]
        return num_cols, txt
    if scenario == "S5":
        txt = [c for c in txt_encoder_cols if "twitter" in c]
        return num_cols, txt
    if scenario == "S7":
        return num_cols, txt_encoder_cols
    if scenario == "S2":
        txt = [c for c in txt_encoder_cols if c.startswith("score_")]
        return num_cols, txt
    if scenario == "S3":
        txt = [c for c in txt_encoder_cols if c.startswith("emb_")]
        return num_cols, txt
    if scenario == "S6":
        txt = [c for c in txt_encoder_cols if ("news" in c or "reddit" in c)]
        return num_cols, txt
    return num_cols, txt_encoder_cols


def _apply_model_ablation_columns(model: str, num_cols: list[str]) -> list[str]:
    cols = list(num_cols)
    if model == "A5":
        cols = [c for c in cols if c.lower() not in {"hashrate", "nvtadj", "nvtadj90", "adractcnt", "txtfrvaladjusd", "mvrv"} and not c.lower().startswith("adr")]
    elif model == "A6":
        cols = [c for c in cols if not c.endswith("_close")]
    return cols


def run_single_config(config_like: dict) -> dict[str, pd.DataFrame]:
    cfg = RunConfig(**config_like)
    lock_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = _load_frozen(cfg.data_version)
    num_cols, txt_cols = _scenario_columns(df, cfg.input_scenario, cfg.text_encoder)
    num_cols = _apply_model_ablation_columns(cfg.model, num_cols)
    folds = make_5fold_tss(df)
    fold = [f for f in folds if f.fold == cfg.fold][0]

    train_df = df[fold.train_mask].copy()
    val_df = df[fold.val_mask].copy()
    test_df = df[fold.test_mask].copy()

    target_col = f"target_logret_t{cfg.horizon}"
    vol_col = f"target_vol_gk_t{cfg.horizon}"
    cls_col = f"target_dir_t{cfg.horizon}"

    if cfg.model in {"M1", "M2", "M3"}:
        pred_reg = run_classic_model(cfg.model, train_df, test_df, num_cols, target_col)
        pred_cls = (pred_reg > 0).astype(float)
        pred_vol = np.abs(pred_reg)
        pred_df = pd.DataFrame(
            {
                "date": test_df["date"].dt.strftime("%Y-%m-%d").values,
                "y_true_reg": test_df[target_col].values,
                "y_true_cls": test_df[cls_col].values,
                "y_true_vol": test_df[vol_col].values,
                "y_pred_reg": pred_reg,
                "y_pred_cls_prob": pred_cls,
                "y_pred_vol": pred_vol,
            }
        )
        att_df = pd.DataFrame({"date": pred_df["date"], "alpha_t": np.nan})
        best_val = np.nan
    else:
        train_w = build_windows(train_df, num_cols, txt_cols, cfg.lookback, cfg.horizon)
        val_w = build_windows(val_df, num_cols, txt_cols, cfg.lookback, cfg.horizon)
        test_w = build_windows(test_df, num_cols, txt_cols, cfg.lookback, cfg.horizon)
        model = build_neural_model(
            cfg.model,
            num_dim=train_w.x_num.shape[-1],
            txt_dim=max(train_w.x_txt.shape[-1], 1),
            hidden_dim=int(cfg.hparams.get("hidden_dim", 128)),
            n_heads=int(cfg.hparams.get("n_heads", 4)),
        )
        out = fit_predict_model(
            model=model,
            train_data=train_w,
            val_data=val_w,
            test_data=test_w,
            seed=cfg.seed,
            lr=float(cfg.hparam("lr", 1e-4)),
            batch_size=int(cfg.hparam("batch_size", 64)),
            max_epoch=int(cfg.hparam("max_epoch", 200)),
            patience=int(cfg.hparam("patience", 20)),
            warmup_epochs=int(cfg.hparam("warmup_epochs", 8)),
            staged_training=bool(cfg.hparam("staged_training", True)),
            stage_switch_epoch=int(cfg.hparam("stage_switch_epoch", 10)),
            device=device,
        )
        pred_df = out.predictions
        pred_df["horizon"] = cfg.horizon
        att_df = out.attention
        trace_df = out.trace
        best_val = float(out.metrics["best_val_loss"].iloc[0])
    if cfg.model in {"M1", "M2", "M3"}:
        trace_df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "alpha_mean", "lr", "stage_switched"])

    y_true_reg = pred_df["y_true_reg"].values
    y_pred_reg = pred_df["y_pred_reg"].values
    y_true_cls = pred_df["y_true_cls"].values
    y_pred_cls = pred_df["y_pred_cls_prob"].values
    y_true_vol = pred_df["y_true_vol"].values
    y_pred_vol = pred_df["y_pred_vol"].values

    reg_m = regression_metrics(y_true_reg, y_pred_reg)
    cls_m = classification_metrics(y_true_cls, y_pred_cls)
    vol_m = volatility_metrics(np.clip(y_true_vol, 1e-8, None), np.clip(y_pred_vol, 1e-8, None))
    rv_proxy = np.clip(pred_df["y_true_vol"].values, 1e-8, None)
    mean_rv20 = float(np.nanmean(np.clip(train_df[vol_col].values, 1e-8, None)))
    bt = run_backtest(
        y_true_reg,
        y_pred_cls,
        fee_bps=cfg.backtest["fee_bps"],
        slippage_bps=cfg.backtest["slippage_bps"],
        rv20=rv_proxy,
        mean_rv20=mean_rv20,
    )

    metrics = {"config_id": cfg.config_id, "best_val_loss": best_val, **reg_m, **cls_m, **vol_m, **bt}
    metrics_df = pd.DataFrame([metrics])

    return {
        "predictions": pred_df,
        "metrics": metrics_df,
        "attention": att_df,
        "trace": trace_df,
    }

