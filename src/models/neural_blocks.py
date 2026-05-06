from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.reg = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, 1))
        self.cls = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, 1), nn.Sigmoid())
        self.vol = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"reg": self.reg(x), "cls": self.cls(x), "vol": self.vol(x)}


class NumericLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out

