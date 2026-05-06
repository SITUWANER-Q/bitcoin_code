from __future__ import annotations

import torch
import torch.nn as nn

from src.models.neural_blocks import MultiTaskHead, NumericLSTMEncoder


class LSTMOnlyModel(nn.Module):
    def __init__(self, num_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.num_encoder = NumericLSTMEncoder(num_dim, hidden_dim=hidden_dim)
        self.head = MultiTaskHead(hidden_dim)

    def forward(self, x_num: torch.Tensor, x_txt: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.num_encoder(x_num)[:, -1, :]
        out = self.head(h)
        out["alpha"] = torch.full((x_num.shape[0], 1), 0.5, device=x_num.device)
        return out


class GRUOnlyModel(nn.Module):
    def __init__(self, num_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(num_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.head = MultiTaskHead(hidden_dim)

    def forward(self, x_num: torch.Tensor, x_txt: torch.Tensor) -> dict[str, torch.Tensor]:
        out, _ = self.gru(x_num)
        h = out[:, -1, :]
        y = self.head(h)
        y["alpha"] = torch.full((x_num.shape[0], 1), 0.5, device=x_num.device)
        return y


class TransformerOnlyModel(nn.Module):
    def __init__(self, num_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(num_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = MultiTaskHead(hidden_dim)

    def forward(self, x_num: torch.Tensor, x_txt: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(self.proj(x_num))[:, -1, :]
        out = self.head(h)
        out["alpha"] = torch.full((x_num.shape[0], 1), 0.5, device=x_num.device)
        return out


class EarlyFusionModel(nn.Module):
    def __init__(self, num_dim: int, txt_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(num_dim + txt_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.head = MultiTaskHead(hidden_dim)

    def forward(self, x_num: torch.Tensor, x_txt: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([x_num, x_txt], dim=-1)
        out, _ = self.lstm(torch.relu(self.proj(x)))
        h = out[:, -1, :]
        y = self.head(h)
        y["alpha"] = torch.full((x_num.shape[0], 1), 0.5, device=x_num.device)
        return y


class LateFusionModel(nn.Module):
    def __init__(self, num_dim: int, txt_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.num_enc = NumericLSTMEncoder(num_dim, hidden_dim=hidden_dim)
        self.txt_enc = NumericLSTMEncoder(txt_dim, hidden_dim=hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.head = MultiTaskHead(hidden_dim)

    def forward(self, x_num: torch.Tensor, x_txt: torch.Tensor) -> dict[str, torch.Tensor]:
        hn = self.num_enc(x_num)[:, -1, :]
        ht = self.txt_enc(x_txt)[:, -1, :]
        alpha = self.gate(torch.cat([hn, ht], dim=-1))
        h = alpha * hn + (1 - alpha) * ht
        out = self.head(h)
        out["alpha"] = alpha
        return out


class GatedFusionModel(nn.Module):
    def __init__(self, num_dim: int, txt_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.num_enc = NumericLSTMEncoder(num_dim, hidden_dim=hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.head = MultiTaskHead(hidden_dim)

    def forward(self, x_num: torch.Tensor, x_txt: torch.Tensor) -> dict[str, torch.Tensor]:
        hn = self.num_enc(x_num)
        ht = torch.relu(self.txt_proj(x_txt))
        g = self.gate(torch.cat([hn, ht], dim=-1))
        h = g * hn + (1 - g) * ht
        pooled = h[:, -1, :]
        out = self.head(pooled)
        out["alpha"] = g[:, -1, :1]
        return out


class CrossAttentionModel(nn.Module):
    def __init__(
        self,
        num_dim: int,
        txt_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        direction: str = "num_to_text",
        use_regime_gate: bool = False,
    ):
        super().__init__()
        self.direction = direction
        self.use_regime_gate = use_regime_gate
        self.num_enc = NumericLSTMEncoder(num_dim, hidden_dim=hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        self.nt_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.tn_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.regime_gate = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.head = MultiTaskHead(hidden_dim)

    def freeze_text_branch(self, freeze: bool) -> None:
        modules = [self.txt_proj, self.nt_attn, self.tn_attn]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = not freeze

    def _regime_features(self, x_num: torch.Tensor) -> torch.Tensor:
        # Use first 4 numeric channels as proxy: rv20/atr/ma_slope/macd_abs from engineered matrix.
        return x_num[:, :, :4]

    def forward(self, x_num: torch.Tensor, x_txt: torch.Tensor) -> dict[str, torch.Tensor]:
        hn = self.num_enc(x_num)
        ht = torch.relu(self.txt_proj(x_txt))

        n_to_t, _ = self.nt_attn(query=hn, key=ht, value=ht)
        t_to_n, _ = self.tn_attn(query=ht, key=hn, value=hn)

        if self.direction == "num_to_text":
            fused = n_to_t + hn
            alpha = torch.ones((x_num.shape[0], 1), device=x_num.device)
        elif self.direction == "text_to_num":
            fused = t_to_n + hn
            alpha = torch.zeros((x_num.shape[0], 1), device=x_num.device)
        else:
            if self.use_regime_gate:
                if x_num.is_cuda:
                    with torch.amp.autocast(device_type="cuda", enabled=False):
                        rg = self._regime_features(x_num)[:, -1, :].float()
                        alpha = self.regime_gate(rg)
                else:
                    rg = self._regime_features(x_num)[:, -1, :].float()
                    alpha = self.regime_gate(rg)
            else:
                alpha = torch.full((x_num.shape[0], 1), 0.5, device=x_num.device)
            fused = alpha.unsqueeze(-1) * n_to_t + (1 - alpha.unsqueeze(-1)) * t_to_n + hn
        out = self.head(fused[:, -1, :])
        out["alpha"] = alpha
        return out

