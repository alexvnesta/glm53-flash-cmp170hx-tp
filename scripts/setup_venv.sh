#!/usr/bin/env bash
# Create .venv with the pinned PyTorch (CUDA 13.0 wheels) and the service dependencies.
# Needs Python 3.13 (the native extension is built for the venv's interpreter) and a CUDA 13.x toolkit with nvcc.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${GLM53_VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3.13}"
command -v "$PYTHON" >/dev/null || { echo "need $PYTHON (set PYTHON=/path/to/python3.13)"; exit 1; }
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip wheel setuptools
"$PY" -m pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
"$PY" -m pip install -r "$ROOT/requirements.txt"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name}, {p.total_memory // 2**20} MiB, sm_{p.major}{p.minor}")
PY
echo "Next: ./engine/apply_engine.sh"
