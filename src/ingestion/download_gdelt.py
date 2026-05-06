from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import requests

from src.config import RAW_DIR


def iter_days(start: str, end: str):
    cur = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    while cur <= end_dt:
        yield cur
        cur += timedelta(days=1)


def download_gdelt(start: str, end: str, keywords: str) -> list[Path]:
    out_dir = RAW_DIR / "news" / "gdelt"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for dt in iter_days(start, end):
        day = dt.strftime("%Y%m%d")
        query = quote_plus(keywords)
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={query}&mode=ArtList&maxrecords=250&format=json&startdatetime={day}000000&enddatetime={day}235959"
        )
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            continue
        out_path = out_dir / f"{day}.json"
        out_path.write_text(resp.text, encoding="utf-8")
        saved.append(out_path)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--keywords", default="bitcoin,btc,cryptocurrency")
    args = parser.parse_args()
    files = download_gdelt(args.start, args.end, args.keywords)
    print(f"[download_gdelt] saved {len(files)} files")

