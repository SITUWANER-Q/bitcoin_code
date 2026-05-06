from __future__ import annotations

import numpy as np
import pandas as pd
import torch.nn as nn

from src.models.fusion_models import (
    CrossAttentionModel,
    EarlyFusionModel,
    GatedFusionModel,
    GRUOnlyModel,
    LSTMOnlyModel,
    LateFusionModel,
    TransformerOnlyModel,
)


def build_neural_model(model_name: str, num_dim: int, txt_dim: int, hidden_dim: int = 128, n_heads: int = 4) -> nn.Module:
    if model_name == "M4":
        return LSTMOnlyModel(num_dim=num_dim, hidden_dim=hidden_dim)
    if model_name == "M5":
        return GRUOnlyModel(num_dim=num_dim, hidden_dim=hidden_dim)
    if model_name == "M6":
        return TransformerOnlyModel(num_dim=num_dim, hidden_dim=hidden_dim)
    if model_name == "M7":
        return EarlyFusionModel(num_dim=num_dim, txt_dim=txt_dim, hidden_dim=hidden_dim)
    if model_name == "M8":
        return LateFusionModel(num_dim=num_dim, txt_dim=txt_dim, hidden_dim=hidden_dim)
    if model_name == "M9":
        return GatedFusionModel(num_dim=num_dim, txt_dim=txt_dim, hidden_dim=hidden_dim)
    if model_name == "M10":
        return CrossAttentionModel(num_dim=num_dim, txt_dim=txt_dim, hidden_dim=hidden_dim, n_heads=n_heads, direction="num_to_text")
    if model_name == "M11":
        return CrossAttentionModel(num_dim=num_dim, txt_dim=txt_dim, hidden_dim=hidden_dim, n_heads=n_heads, direction="text_to_num")
    if model_name in {"M12", "A1", "A2", "A3", "A4", "A7", "A8"}:
        direction = "bi"
        use_gate = model_name != "A3"
        heads = 1 if model_name == "A4" else n_heads
        return CrossAttentionModel(
            num_dim=num_dim,
            txt_dim=txt_dim,
            hidden_dim=hidden_dim,
            n_heads=heads,
            direction=direction,
            use_regime_gate=use_gate,
        )
    raise ValueError(f"unsupported neural model: {model_name}")


def run_classic_model(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> np.ndarray:
    x_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[target_col].values
    x_test = test_df[feature_cols].fillna(0).values

    if model_name == "M1":
        from statsmodels.tsa.arima.model import ARIMA

        # ARIMA on target itself as univariate baseline.
        fit = ARIMA(y_train, order=(2, 1, 2)).fit()
        pred = fit.forecast(steps=len(test_df))
        return np.asarray(pred)
    if model_name == "M2":
        from arch import arch_model

        fit = arch_model(y_train * 100, vol="GARCH", p=1, q=1).fit(disp="off")
        f = fit.forecast(horizon=len(test_df), reindex=False).variance.values[-1]
        # Convert variance forecast to signed return proxy with test mean sign.
        sign = np.sign(np.nanmean(y_train)) if np.nanmean(y_train) != 0 else 1.0
        return sign * np.sqrt(np.clip(f, 0, None)) / 100
    if model_name == "M3":
        from xgboost import XGBRegressor

        reg = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
        )
        reg.fit(x_train, y_train)
        return reg.predict(x_test)
    raise ValueError(f"unsupported classic model: {model_name}")

