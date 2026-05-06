from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.training.dataset import TimeSeriesWindowDataset, WindowedData
from src.training.multitask_loss import MultiTaskLoss
from src.training.seed_utils import worker_init_fn


@dataclass
class TrainOutput:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    attention: pd.DataFrame
    trace: pd.DataFrame


def build_windows(
    df: pd.DataFrame,
    numerical_cols: list[str],
    text_cols: list[str],
    lookback: int,
    horizon: int,
) -> WindowedData:
    num = df[numerical_cols].values
    txt = df[text_cols].values if text_cols else np.zeros((len(df), 1))
    y_reg = df[f"target_logret_t{horizon}"].values
    y_cls = df[f"target_dir_t{horizon}"].values
    y_vol = df[f"target_vol_gk_t{horizon}"].values
    dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").tolist()

    x_num, x_txt, o_reg, o_cls, o_vol, o_dates = [], [], [], [], [], []
    for i in range(lookback, len(df)):
        x_num.append(num[i - lookback : i])
        x_txt.append(txt[i - lookback : i])
        o_reg.append([y_reg[i]])
        o_cls.append([y_cls[i]])
        o_vol.append([y_vol[i]])
        o_dates.append(dates[i])
    return WindowedData(
        x_num=np.array(x_num, dtype=np.float32),
        x_txt=np.array(x_txt, dtype=np.float32),
        y_reg=np.array(o_reg, dtype=np.float32),
        y_cls=np.array(o_cls, dtype=np.float32),
        y_vol=np.array(o_vol, dtype=np.float32),
        dates=o_dates,
    )


def _as_targets(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "y_reg": batch["y_reg"],
        "y_cls": batch["y_cls"],
        "y_vol": batch["y_vol"],
    }


def fit_predict_model(
    model: torch.nn.Module,
    train_data: WindowedData,
    val_data: WindowedData,
    test_data: WindowedData,
    seed: int,
    lr: float = 1e-4,
    batch_size: int = 64,
    max_epoch: int = 200,
    patience: int = 15,
    warmup_epochs: int = 8,
    staged_training: bool = True,
    stage_switch_epoch: int = 10,
    device: str = "cuda",
) -> TrainOutput:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = MultiTaskLoss()
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    loader_workers = 10 if use_cuda else 0
    if use_cuda:
        torch.set_float32_matmul_precision("high")
    scaler = torch.amp.GradScaler("cuda", enabled=False) if use_cuda else None
    warmup_epochs = max(1, int(warmup_epochs))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch_idx: min(1.0, float(epoch_idx + 1) / float(warmup_epochs)),
    )
    staged_enabled = staged_training and hasattr(model, "freeze_text_branch")
    if staged_enabled:
        model.freeze_text_branch(True)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "worker_init_fn": worker_init_fn,
        "num_workers": loader_workers,
        "pin_memory": use_cuda,
    }
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(TimeSeriesWindowDataset(train_data), **loader_kwargs)
    val_loader = DataLoader(TimeSeriesWindowDataset(val_data), **loader_kwargs)
    test_loader = DataLoader(TimeSeriesWindowDataset(test_data), **loader_kwargs)

    best_state = None
    best_val = float("inf")
    no_improve = 0
    trace_rows: list[dict[str, float]] = []
    stage_switched = False

    for epoch_idx in range(max_epoch):
        if staged_enabled and epoch_idx == int(stage_switch_epoch):
            model.freeze_text_branch(False)
            # Soft landing when unfreezing attention branch to avoid sharp oscillation.
            for param_group in optimizer.param_groups:
                param_group["lr"] = float(param_group["lr"]) * 0.5
            stage_switched = True
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        train_alpha_sum = 0.0
        train_alpha_count = 0
        for batch in train_loader:
            for k in ("x_num", "x_txt", "y_reg", "y_cls", "y_vol"):
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            if use_cuda:
                amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                amp_ctx = nullcontext()
            with amp_ctx:
                preds = model(batch["x_num"], batch["x_txt"])
                loss = criterion(preds, _as_targets(batch))
            alpha = preds.get("alpha")
            if alpha is not None:
                alpha_det = alpha.detach().float()
                train_alpha_sum += float(alpha_det.mean().item()) * alpha_det.shape[0]
                train_alpha_count += int(alpha_det.shape[0])
            train_loss_sum += float(loss.item())
            train_count += 1
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                for k in ("x_num", "x_txt", "y_reg", "y_cls", "y_vol"):
                    batch[k] = batch[k].to(device)
                if use_cuda:
                    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                else:
                    amp_ctx = nullcontext()
                with amp_ctx:
                    preds = model(batch["x_num"], batch["x_txt"])
                    loss = criterion(preds, _as_targets(batch))
                val_loss += float(loss.item())
                val_count += 1
        val_loss = val_loss / max(val_count, 1)
        scheduler.step()
        train_loss = train_loss_sum / max(train_count, 1)
        alpha_mean = train_alpha_sum / max(train_alpha_count, 1) if train_alpha_count > 0 else float("nan")
        current_lr = float(optimizer.param_groups[0]["lr"])
        trace_rows.append(
            {
                "epoch": float(epoch_idx),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "alpha_mean": float(alpha_mean),
                "lr": current_lr,
                "stage_switched": 1.0 if (stage_switched and epoch_idx >= int(stage_switch_epoch)) else 0.0,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    rows = []
    att_rows = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x_num = batch["x_num"].to(device)
            x_txt = batch["x_txt"].to(device)
            if use_cuda:
                amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                amp_ctx = nullcontext()
            with amp_ctx:
                out = model(x_num, x_txt)
            reg = out["reg"].detach().cpu().numpy().reshape(-1)
            cls = out["cls"].detach().cpu().numpy().reshape(-1)
            vol = out["vol"].detach().cpu().numpy().reshape(-1)
            alpha = out.get("alpha")
            alpha_np = alpha.detach().cpu().numpy().reshape(-1) if alpha is not None else np.full_like(reg, np.nan)
            for i, date in enumerate(batch["date"]):
                rows.append(
                    {
                        "date": date,
                        "y_true_reg": float(batch["y_reg"][i].item()),
                        "y_true_cls": float(batch["y_cls"][i].item()),
                        "y_true_vol": float(batch["y_vol"][i].item()),
                        "y_pred_reg": float(reg[i]),
                        "y_pred_cls_prob": float(cls[i]),
                        "y_pred_vol": float(vol[i]),
                    }
                )
                att_rows.append({"date": date, "alpha_t": float(alpha_np[i])})

    pred_df = pd.DataFrame(rows)
    metrics = pd.DataFrame(
        [
            {
                "best_val_loss": best_val,
                "seed": seed,
                "stage_switch_epoch": int(stage_switch_epoch),
                "staged_training": int(staged_enabled),
            }
        ]
    )
    att_df = pd.DataFrame(att_rows)
    trace_df = pd.DataFrame(trace_rows)
    return TrainOutput(pred_df, metrics, att_df, trace_df)

