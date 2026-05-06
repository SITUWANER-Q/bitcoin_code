from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    config_id: str
    group: str
    model: str
    input_scenario: str
    horizon: int
    lookback: int
    text_encoder: str
    seed: int
    fold: int
    data_version: str
    hparams: dict[str, Any]
    loss_weights: dict[str, float]
    backtest: dict[str, float]
    output_dir: str

    @staticmethod
    def from_yaml(path: Path) -> "RunConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return RunConfig(**raw)

    def hparam(self, key: str, default: Any) -> Any:
        return self.hparams.get(key, default)

