from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


ENCODER_MAP = {
    "E1": "ElKulako/cryptobert",
    "E2": "yiyanghkust/finbert-tone",
    "E3": "sentence-transformers/all-MiniLM-L6-v2",
}


@dataclass
class EncodedBatch:
    scores: np.ndarray
    embeddings: np.ndarray


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_SPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    s = str(text).lower()
    s = _URL_RE.sub(" ", s)
    s = _MENTION_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    if len(s) > 1200:
        s = s[:1200]
    return s


def _parallel_clean(texts: list[str], n_jobs: int) -> list[str]:
    return Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_clean_text)(t) for t in texts
    )


class SentimentEncoder:
    def __init__(self, encoder_id: str, device: str = "cuda"):
        self.encoder_id = encoder_id
        model_name = ENCODER_MAP[encoder_id]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.is_classifier = encoder_id in {"E1", "E2"}
        if self.is_classifier:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, output_hidden_states=True
            ).to(device)
        else:
            self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> EncodedBatch:
        if not texts:
            return EncodedBatch(np.zeros((0,)), np.zeros((0, 768)))

        all_scores = []
        all_embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            tokens = self.tokenizer(
                chunk, truncation=True, padding=True, max_length=128, return_tensors="pt"
            ).to(self.device)
            out = self.model(**tokens)
            hidden = out.hidden_states[-1].mean(dim=1)
            if self.is_classifier:
                logits = out.logits
                probs = torch.softmax(logits, dim=-1)
                if probs.shape[1] >= 3:
                    score = probs[:, 2] - probs[:, 0]
                else:
                    score = probs[:, 1] - probs[:, 0]
            else:
                score = torch.zeros(hidden.shape[0], device=hidden.device)
            all_scores.append(score.detach().cpu().numpy())
            all_embs.append(hidden.detach().cpu().numpy())
        return EncodedBatch(
            np.concatenate(all_scores, axis=0),
            np.concatenate(all_embs, axis=0),
        )


def aggregate_daily_sentiment(
    df: pd.DataFrame,
    encoder: SentimentEncoder,
    source_name: str,
    n_jobs: int = -1,
) -> pd.DataFrame:
    work = df.copy()
    work = work.dropna(subset=["date"])
    # Enforce UTC normalization before daily aggregation to avoid timezone leakage.
    work["date"] = pd.to_datetime(work["date"], utc=True, errors="coerce")
    work = work.dropna(subset=["date"])
    work["date"] = work["date"].dt.date.astype(str)
    work["text"] = work["text"].astype(str).fillna("")
    work["text"] = _parallel_clean(work["text"].tolist(), n_jobs=n_jobs)
    grouped = work.groupby("date")["text"].apply(list).reset_index(name="texts")
    rows = []
    for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc=f"sentiment:{source_name}"):
        encoded = encoder.encode_texts(row["texts"])
        if encoded.scores.size == 0:
            continue
        emb = encoded.embeddings.mean(axis=0)
        rows.append(
            {
                "date": row["date"],
                f"score_{source_name}": float(encoded.scores.mean()),
                f"score_std_{source_name}": float(encoded.scores.std()),
                f"count_{source_name}": int(encoded.scores.size),
                **{f"emb_{source_name}_{i:03d}": float(v) for i, v in enumerate(emb)},
            }
        )
    return pd.DataFrame(rows)

