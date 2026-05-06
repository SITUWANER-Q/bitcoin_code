from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.config import FROZEN_DIR, TARGET_END_DATE, ensure_dirs
from src.features.pipeline import build_aligned_dataset
from src.utils.io import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--encoders", nargs="+", default=["E1", "E2", "E3"])
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    ensure_dirs()
    out_dir = FROZEN_DIR / args.version
    out_dir.mkdir(parents=True, exist_ok=True)

    aligned = build_aligned_dataset(
        encoders=args.encoders,
        max_docs_per_source=args.max_docs_per_source,
        n_jobs=args.n_jobs,
    )
    full = aligned.full

    numerical_path = out_dir / "numerical_daily.parquet"
    sentiment_path = out_dir / "sentiment_daily.parquet"
    labels_path = out_dir / "labels_daily.parquet"

    numerical_df = full[["date", *aligned.numerical_cols]]
    sentiment_df = full[["date", *aligned.text_cols]]
    labels_df = full[["date", *aligned.label_cols]]

    numerical_df.to_parquet(numerical_path, index=False)
    sentiment_df.to_parquet(sentiment_path, index=False)
    labels_df.to_parquet(labels_path, index=False)

    manifest = {
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_end_date": TARGET_END_DATE,
        "encoders": args.encoders,
        "numerical_file": str(numerical_path.relative_to(out_dir)),
        "sentiment_file": str(sentiment_path.relative_to(out_dir)),
        "labels_file": str(labels_path.relative_to(out_dir)),
        "sha256": {
            "numerical_daily.parquet": sha256_file(numerical_path),
            "sentiment_daily.parquet": sha256_file(sentiment_path),
            "labels_daily.parquet": sha256_file(labels_path),
        },
        "columns": {
            "numerical": aligned.numerical_cols,
            "text": aligned.text_cols,
            "labels": aligned.label_cols,
        },
        "n_rows": int(full.shape[0]),
    }
    write_json(out_dir / "MANIFEST.json", manifest)
    print(f"[freeze_data] done: {out_dir}")


if __name__ == "__main__":
    main()

