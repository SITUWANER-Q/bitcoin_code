from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, w_reg: float = 1.0, w_cls: float = 1.0, w_vol: float = 0.5):
        super().__init__()
        self.w_reg = w_reg
        self.w_cls = w_cls
        self.w_vol = w_vol
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, preds: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> torch.Tensor:
        reg = self.mse(preds["reg"], target["y_reg"])
        cls = self.bce(preds["cls"], target["y_cls"])
        vol = self.mse(preds["vol"], target["y_vol"])
        return self.w_reg * reg + self.w_cls * cls + self.w_vol * vol

