from __future__ import annotations

import argparse
import json
import os
import signal
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
import torch.multiprocessing as mp
from tqdm import tqdm

from src.config import CONFIGS_DIR, LOGS_DIR, RESULTS_DIR
from src.runners.train_one import run_single_config


INTERRUPTED = False


def _trap(signum, frame):  # noqa: ANN001
    global INTERRUPTED
    INTERRUPTED = True
    print(f"\n[run_batch] signal={signum}, stopping after current jobs")
    # Defensive cleanup for persistent DataLoader workers on abrupt interrupt.
    if os.name == "posix":
        os.system("pkill -9 -f train_one.py >/dev/null 2>&1")


def _run_one(cfg_path: Path, out_root: Path, logs_root: Path, gpu_id: int, max_retry: int) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out_dir = out_root / cfg_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_root / f"{cfg_path.stem}.log"

    for attempt in range(1, max_retry + 1):
        try:
            result = run_single_config(cfg)
            result["predictions"].to_parquet(out_dir / "predictions.parquet", index=False)
            result["metrics"].to_parquet(out_dir / "metrics.parquet", index=False)
            result["attention"].to_parquet(out_dir / "attention.parquet", index=False)
            if "trace" in result and isinstance(result["trace"], pd.DataFrame):
                result["trace"].to_parquet(out_dir / "trace.parquet", index=False)
            meta = {
                "config_id": cfg["config_id"],
                "attempt": attempt,
                "group": cfg["group"],
                "model": cfg["model"],
                "data_version": cfg["data_version"],
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            (out_dir / "_DONE").touch()
            return {"id": cfg_path.stem, "ok": True}
        except Exception as exc:  # noqa: BLE001
            log_path.write_text(f"attempt={attempt}\n{traceback.format_exc()}\n", encoding="utf-8")
            if attempt == max_retry:
                return {"id": cfg_path.stem, "ok": False, "err": str(exc)}
    return {"id": cfg_path.stem, "ok": False, "err": "unexpected"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("group", nargs="?", default=None)
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--max-retry", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    signal.signal(signal.SIGINT, _trap)
    signal.signal(signal.SIGTERM, _trap)

    cfg_dir = CONFIGS_DIR / args.version
    result_root = RESULTS_DIR / args.version
    logs_root = LOGS_DIR / args.version
    result_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    all_cfg = sorted(cfg_dir.glob("*.yaml"))
    if args.group:
        all_cfg = [p for p in all_cfg if p.name.startswith(f"{args.group}__")]

    done = {p.parent.name for p in result_root.glob("*/_DONE")}
    pending = [p for p in all_cfg if p.stem not in done]
    if args.smoke:
        pending = pending[:5]

    print(f"[run_batch] total={len(all_cfg)} done={len(done)} pending={len(pending)}")
    if not pending:
        return

    if args.n_gpus <= 1:
        for cfg_path in tqdm(pending, desc="batch"):
            if INTERRUPTED:
                break
            r = _run_one(cfg_path, result_root, logs_root, gpu_id=0, max_retry=args.max_retry)
            print(f"[{'OK' if r['ok'] else 'FAIL'}] {r['id']}")
    else:
        with ProcessPoolExecutor(max_workers=args.n_gpus) as ex:
            futures = {
                ex.submit(_run_one, cfg, result_root, logs_root, i % args.n_gpus, args.max_retry): cfg
                for i, cfg in enumerate(pending)
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="batch"):
                if INTERRUPTED:
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
                r = fut.result()
                print(f"[{'OK' if r['ok'] else 'FAIL'}] {r['id']}")

    print("[run_batch] done")


if __name__ == "__main__":
    main()

