from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import requests

from src.config import RAW_DIR

GDELT_KEYWORDS = "(bitcoin OR btc OR cryptocurrency)"


def _is_valid_gdelt_payload(text: str) -> bool:
    if "illegal character" in text.lower():
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(payload.get("articles"), list)


def _is_valid_cached(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        return _is_valid_gdelt_payload(path.read_text(encoding="utf-8"))
    except OSError:
        return False


def iter_days(start: str, end: str):
    cur = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    while cur <= end_dt:
        yield cur
        cur += timedelta(days=1)


def download_gdelt(start: str, end: str, keywords: str) -> list[Path]:
    """Download GDELT daily article lists. Existing non-empty json files are kept (per-day resume)."""
    out_dir = RAW_DIR / "news" / "gdelt"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    skipped = 0
    for dt in iter_days(start, end):
        day = dt.strftime("%Y%m%d")
        out_path = out_dir / f"{day}.json"
        if _is_valid_cached(out_path):
            saved.append(out_path)
            skipped += 1
            continue
        query = quote_plus(keywords)
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={query}&mode=ArtList&maxrecords=250&format=json&startdatetime={day}000000&enddatetime={day}235959"
        )
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            continue
        if not _is_valid_gdelt_payload(resp.text):
            continue
        out_path.write_text(resp.text, encoding="utf-8")
        saved.append(out_path)
    if skipped:
        print(f"[download_gdelt] kept {skipped} existing json files")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--keywords", default=GDELT_KEYWORDS)
    args = parser.parse_args()
    files = download_gdelt(args.start, args.end, args.keywords)
    print(f"[download_gdelt] saved {len(files)} files")

