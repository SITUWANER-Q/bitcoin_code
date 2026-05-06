from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from src.config import CONFIGS_DIR, ensure_dirs


def build_config(
    group: str,
    model: str,
    scenario: str,
    horizon: int,
    lookback: int,
    encoder: str,
    seed: int,
    fold: int,
    data_version: str = "v1.0.0",
) -> dict:
    config_id = f"{group}__{model}__{scenario}__H{horizon}__W{lookback}__{encoder}__seed{seed}__fold{fold}"
    return {
        "config_id": config_id,
        "group": group,
        "model": model,
        "input_scenario": scenario,
        "horizon": horizon,
        "lookback": lookback,
        "text_encoder": encoder,
        "seed": seed,
        "fold": fold,
        "data_version": data_version,
        "hparams": {
            "hidden_dim": 128,
            "n_layers": 2,
            "n_heads": 4,
            "dropout": 0.1,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "batch_size": 64,
            "max_epoch": 200,
            "patience": 20,
            "warmup_epochs": 8,
            "staged_training": True,
            "stage_switch_epoch": 10,
        },
        "loss_weights": {"reg": 1.0, "cls": 1.0, "vol": 0.5},
        "backtest": {"fee_bps": 10.0, "slippage_bps": 5.0},
        "output_dir": f"results/{data_version}/{config_id}",
    }


def main() -> None:
    ensure_dirs()
    version = "v1.0.0"
    out_dir = CONFIGS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [2024, 2025, 42]
    folds = [1, 2, 3, 4, 5]
    models = [f"M{i}" for i in range(1, 13)]
    main_scenarios = ["S1", "S4", "S5", "S7"]

    configs = []
    for m in models:
        for s in main_scenarios:
            for seed in seeds:
                for fold in folds:
                    cfg = build_config("main", m, s, 1, 14, "E1", seed, fold, version)
                    configs.append(cfg)

    # robustness (H={1,3}, W={14,30}) for M12+S7+E1
    for h in [1, 3]:
        for w in [14, 30]:
            if h == 1 and w == 14:
                continue
            for seed in seeds:
                for fold in folds:
                    configs.append(build_config("robustness", "M12", "S7", h, w, "E1", seed, fold, version))

    # encoder ablation (E2/E3)
    for e in ["E2", "E3"]:
        for seed in seeds:
            for fold in folds:
                configs.append(build_config("encoder", "M12", "S7", 1, 14, e, seed, fold, version))

    # component ablation A1-A8
    for a in [f"A{i}" for i in range(1, 9)]:
        for seed in seeds:
            for fold in folds:
                configs.append(build_config("ablation", a, "S7", 1, 14, "E1", seed, fold, version))

    # appendix S2/S3/S6 only fold3/seed2024
    for m in models:
        for s in ["S2", "S3", "S6"]:
            configs.append(build_config("appendix", m, s, 1, 14, "E1", 2024, 3, version))

    # noise robustness dropout 20/50 via model tags
    for suffix in ["N20", "N50"]:
        for seed in seeds:
            for fold in folds:
                cfg = build_config("noise", "M12", "S7", 1, 14, "E1", seed, fold, version)
                cfg["hparams"]["text_dropout_tag"] = suffix
                cfg["config_id"] = cfg["config_id"].replace("__fold", f"__{suffix}__fold")
                cfg["output_dir"] = f"results/{version}/{cfg['config_id']}"
                configs.append(cfg)

    rows = []
    for cfg in configs:
        path = out_dir / f"{cfg['config_id']}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        rows.append(
            {
                "config_id": cfg["config_id"],
                "group": cfg["group"],
                "model": cfg["model"],
                "scenario": cfg["input_scenario"],
                "horizon": cfg["horizon"],
                "lookback": cfg["lookback"],
                "encoder": cfg["text_encoder"],
                "seed": cfg["seed"],
                "fold": cfg["fold"],
            }
        )

    manifest = pd.DataFrame(rows).sort_values("config_id")
    manifest.to_csv(out_dir / "MANIFEST_configs.csv", index=False, encoding="utf-8")
    print(f"[generate_configs] total={len(configs)} -> {out_dir}")


if __name__ == "__main__":
    main()

