#!/usr/bin/env bash
# Long-running GDELT downloader with single-instance lock and auto-retry.
# Usage: bash scripts/gdelt_loop.sh
# Background: nohup bash scripts/gdelt_loop.sh >> ~/gdelt_download.log 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GDELT_DIR="${ROOT}/data/raw/news/gdelt"
LOCK_FILE="${GDELT_DIR}/.gdelt_loop.lock"
KEYWORDS="(bitcoin OR btc OR cryptocurrency)"
START_DATE="2018-01-01"
END_DATE="2026-04-30"

mkdir -p "${GDELT_DIR}"

# Only one loop at a time (prevents duplicate nohup / double bash).
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[gdelt_loop] already running (lock: ${LOCK_FILE}). Not starting a second copy."
  exit 1
fi

cd "${ROOT}"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "btc" ]; then
  for _conda_sh in \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "${_conda_sh}" ]; then
      # shellcheck source=/dev/null
      source "${_conda_sh}"
      conda activate btc
      break
    fi
  done
fi

echo "[gdelt_loop] started $(date -Iseconds) pid=$$ keywords=${KEYWORDS}"

while true; do
  # Blocking: only one download_gdelt.py child at a time (no trailing &).
  python -m src.ingestion.download_gdelt \
    --start "${START_DATE}" \
    --end "${END_DATE}" \
    --keywords "${KEYWORDS}" || true

  echo "[gdelt_loop] $(date -Iseconds) download round finished"

  python - <<'PY'
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

start = datetime.fromisoformat("2018-01-01")
end = datetime.fromisoformat("2026-04-30")
gd = Path("data/raw/news/gdelt")

def valid_day(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload.get("articles"), list)

missing = 0
invalid = 0
for d in range((end - start).days + 1):
    day = (start + timedelta(days=d)).strftime("%Y%m%d")
    p = gd / f"{day}.json"
    if not p.exists() or p.stat().st_size == 0:
        missing += 1
    elif not valid_day(p):
        invalid += 1

print(f"[gdelt_loop] missing_days={missing} invalid_days={invalid}", flush=True)
sys.exit(0 if missing == 0 and invalid == 0 else 1)
PY

  if [ $? -eq 0 ]; then
    echo "[gdelt_loop] complete: all days have valid articles JSON"
    break
  fi

  echo "[gdelt_loop] retry in 60s"
  sleep 60
done
