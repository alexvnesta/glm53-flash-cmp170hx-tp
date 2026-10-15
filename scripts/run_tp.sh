#!/usr/bin/env bash
# Start the GLM-5.3-Flash API with the measured production configuration: TP=2 + DFlash2 K7 + hybrid drafting, two concurrent
# sessions, shared prefix cache, 8 GiB RAM tier, 392,960-token context, image input, NCCL (P2P) tensor-parallel backend. Extra arguments are passed to glm_api.py.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${GLM53_VENV:-$ROOT/.venv}/bin/python"
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-8012}"
VISION="${VISION:-1}"; HYBRID_DRAFT_TOKENS="${HYBRID_DRAFT_TOKENS:-15}"; RAM_TIER_GIB="${RAM_TIER_GIB:-8}"
TP_BACKEND="${TP_BACKEND:-nccl}"   # nccl = GPU P2P all-reduce (+4-6 % decode, +10 % prefill here); native = through host RAM
ENGINE_SRC="${GLM53_ENGINE_SRC:-$ROOT/engine-src}"
NATIVE_DIR="${GLM53_NATIVE_DIR:-$ROOT/native}"
for f in "$ENGINE_SRC/exllamav3/__init__.py" "$MODELS_DIR/GLM-5.3-Flash-exl3-3.05bpw/config.json" \
         "$MODELS_DIR/GLM-5.3-Flash-DFlash2-exl3-6bpw/model.safetensors"; do
  [ -e "$f" ] || { echo "missing $f (run apply_engine.sh, build_native.sh and get_models.sh first)"; exit 1; }
done
ls "$NATIVE_DIR"/exllamav3_ext*.so >/dev/null 2>&1 || { echo "missing $NATIVE_DIR/exllamav3_ext*.so (run ./engine/build_native.sh)"; exit 1; }

shopt -s nullglob
EXT=("$NATIVE_DIR"/exllamav3_ext*.so)
[ "${#EXT[@]}" = 1 ] || { echo "expected exactly one native extension" >&2; exit 1; }
if command -v sha256sum >/dev/null; then GLM53_NATIVE_EXT_SHA256=$(sha256sum "${EXT[0]}" | cut -d' ' -f1)
else GLM53_NATIVE_EXT_SHA256=$(shasum -a 256 "${EXT[0]}" | cut -d' ' -f1); fi
export GLM53_NATIVE_EXT_SHA256
export PYTHONPATH="$NATIVE_DIR:$ENGINE_SRC:$ROOT/service/glm${GLM53_EXTRA_PYTHONPATH:+:$GLM53_EXTRA_PYTHONPATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" TORCH_CUDA_ARCH_LIST=8.0
export EXL3_BC_ATTN=1 EXL3_MLA_KIDX_RING=32 EXL3_INT8_GEMV=0 EXL3_GEMV=0 EXL3_DSA_QC_STAGE=1 EXL3_DSA_QC_STAGE_MIN_R=9
export EXL3_TP_WORKER_TIMEOUT=300 EXL3_TP_SHARING_STRATEGY=file_descriptor EXL3_AUTOSPLIT_MARGIN_MB=128
export GLM53_PREFILL_CHUNK_SIZE=2048 GLM53_DYNAMIC_DRAFT=0 GLM53_RECORD_DRAFT_STATS=1 GLM53_TP_ACTIVE_HOST=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"   # NCCL bootstrap stays on loopback
ulimit -n 65536 2>/dev/null || true

ARGS=(-m "$MODELS_DIR/GLM-5.3-Flash-exl3-3.05bpw" -dm "$MODELS_DIR/GLM-5.3-Flash-DFlash2-exl3-6bpw"
  -cq 8 -cs 655360 --context-cap 392960 -ndt 7 -chunk_size 2048 -ambs 2 -gs 62,63 -lv -tp --experimental-dflash2-tp
  -tpb "$TP_BACKEND" --q8-staging --q8-head-gpu0
  --experimental-concurrent-generation --concurrent-max-sequences 2 --concurrent-fair-gen-rounds 4 --concurrent-fair-chunk-size 512
  --host "$HOST" --port "$PORT"
  --experimental-b2-session-cache --b2-target-cpu-cache-gib "$RAM_TIER_GIB" --recurrent-cache-gib 5
  --experimental-b2-shared-prefix)
if [ "$VISION" = 1 ]; then
  SANDBOX=bwrap; command -v bwrap >/dev/null || { echo "vision requires bubblewrap (bwrap); install it or set VISION=0" >&2; exit 1; }
  ARGS+=(--experimental-vision --vision-decode-sandbox "$SANDBOX" --vision-max-image-tokens 6144 --vision-cache-mib 4096
    --vision-max-images 16 --vision-max-request-image-bytes 67108864 --vision-device 0)
fi
[ "$HYBRID_DRAFT_TOKENS" != 0 ] && ARGS+=(--hybrid-draft-tokens "$HYBRID_DRAFT_TOKENS")
exec "$PY" -u "$ROOT/service/glm/glm_api.py" "${ARGS[@]}" "$@"
