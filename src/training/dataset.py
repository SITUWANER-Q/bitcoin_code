from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WindowedData:
    x_num: np.ndarray
    x_txt: np.ndarray
    y_reg: np.ndarray
    y_cls: np.ndarray
    y_vol: np.ndarray
    dates: list[str]


class TimeSeriesWindowDataset(Dataset):
    def __init__(self, data: WindowedData):
        self.data = data

    def __len__(self) -> int:
        return self.data.x_num.shape[0]

    def __getitem__(self, idx: int):
        return {
            "x_num": torch.tensor(self.data.x_num[idx], dtype=torch.float32),
            "x_txt": torch.tensor(self.data.x_txt[idx], dtype=torch.float32),
            "y_reg": torch.tensor(self.data.y_reg[idx], dtype=torch.float32),
            "y_cls": torch.tensor(self.data.y_cls[idx], dtype=torch.float32),
            "y_vol": torch.tensor(self.data.y_vol[idx], dtype=torch.float32),
            "date": self.data.dates[idx],
        }

