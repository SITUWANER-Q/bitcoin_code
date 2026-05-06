from __future__ import annotations

from src.models.factory import build_neural_model


def build(num_dim: int, txt_dim: int, hidden_dim: int = 128, n_heads: int = 4):
    return build_neural_model("A2", num_dim=num_dim, txt_dim=txt_dim, hidden_dim=hidden_dim, n_heads=n_heads)

