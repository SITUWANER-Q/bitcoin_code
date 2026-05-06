from __future__ import annotations

from src.models.factory import run_classic_model


def predict_garch(train_df, test_df, feature_cols, target_col):
    return run_classic_model("M2", train_df, test_df, feature_cols, target_col)

