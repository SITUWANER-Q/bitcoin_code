#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-btc}"
PYTHON_VERSION="${2:-3.10}"

echo "[setup] target host recommendation for Ubuntu 24.04 + RTX 4090:"
echo "[setup]  - install NVIDIA driver via: sudo ubuntu-drivers install (expect 550/555 series)"
echo "[setup]  - keep CUDA runtime at 12.1 for stable PyTorch cu121 wheels"
echo "[setup]  - if extension build fails under GCC 13, fallback:"
echo "[setup]    sudo apt install -y gcc-12 g++-12"
echo "[setup]    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120"
echo "[setup]    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 120"

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup] conda is required but not found."
  exit 1
fi

if command -v gcc >/dev/null 2>&1; then
  GCC_MAJOR="$(gcc -dumpversion | cut -d. -f1 || true)"
  if [[ -n "${GCC_MAJOR}" ]] && [[ "${GCC_MAJOR}" -ge 13 ]]; then
    echo "[setup] detected GCC ${GCC_MAJOR}; for CUDA-extension compile issues, consider gcc-12 fallback."
  fi
fi

echo "[setup] creating/updating conda env: ${ENV_NAME}"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] env exists, skipping create."
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-cu121.txt
python -m pip install -e .

python - <<'PY'
import torch
print("[setup] CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[setup] Device:", torch.cuda.get_device_name(0))
    print("[setup] Torch CUDA runtime:", torch.version.cuda)
    print("[setup] BF16 supported:", torch.cuda.is_bf16_supported())
PY

echo "[setup] optional monitor: sudo apt install -y nvtop"
echo "[setup] done."
