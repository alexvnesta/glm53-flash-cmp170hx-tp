#!/usr/bin/env bash
# Download the target model and the DFlash2 drafter at the pinned revisions, then convert the drafter to EXL3 6 bpw.
# The drafter is CC-BY-NC-ND-4.0: convert it yourself, and don't redistribute the converted weights.
# Conversion uses GPU 0 for a while; run this before starting the server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${GLM53_VENV:-$ROOT/.venv}/bin/python"
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
TARGET_REPO=turboderp/GLM-5.3-Flash-exl3
TARGET_REV=332ab457b709b7ba30dd9a448be5de03b80a7ac9      # 3.05 bpw branch tip
DRAFT_REPO=incoai/GLM-5.3-Flash-DFlash2
DRAFT_REV=bf582e4eacc1810f76656d1811693ff6c6737d2a
export PYTHONPATH="$ROOT/native:$ROOT/engine-src:$ROOT/service/glm${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$MODELS_DIR"
fetch() {  # repo revision dir
  [ -f "$3/config.json" ] && { echo "== $1 already in $3"; return; }
  echo "== downloading $1@$2 -> $3"
  "$PY" -c "import sys; from huggingface_hub import snapshot_download as s; s(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])" "$1" "$2" "$3"
}
fetch "$TARGET_REPO" "$TARGET_REV" "$MODELS_DIR/GLM-5.3-Flash-exl3-3.05bpw"
fetch "$DRAFT_REPO" "$DRAFT_REV" "$MODELS_DIR/GLM-5.3-Flash-DFlash2-bf16"
OUT="$MODELS_DIR/GLM-5.3-Flash-DFlash2-exl3-6bpw"
if [ -f "$OUT/model.safetensors" ]; then echo "== drafter already converted in $OUT"
else
  echo "== converting the drafter to EXL3 6 bpw (tens of minutes, GPU 0)"
  "$PY" "$ROOT/service/dflash2/convert_drafter.py" -i "$MODELS_DIR/GLM-5.3-Flash-DFlash2-bf16" -o "$OUT" \
    -w "$MODELS_DIR/dflash2-quant-work" -b 6 --devices 0
fi
echo "Models ready in $MODELS_DIR. Next: ./scripts/run_tp.sh"
