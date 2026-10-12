#!/usr/bin/env bash
# Build the exllamav3_ext native extension (CUDA sm_80) from the patched engine-src into native/.
#
# This is the recipe of the measured setup: ExLlamaV3's own setup.py (setuptools CUDAExtension; the sources and the
# nvcc flags come from exllamav3/util/cuda_flags.py: -O3 --use_fast_math --compress-mode=size, host C++ -Ofast),
# TORCH_CUDA_ARCH_LIST=8.0, CUDA toolkit 13.3, torch 2.14.1+cu130, MAX_JOBS=4, GPUs hidden, low priority.
# It takes about 25 minutes the first time and compiles on the CPU only; it never loads the result.
#
# The service imports the extension from native/ (PYTHONPATH=native:engine-src:service/glm). ext.py of the patched
# engine refuses an extension without the glm_replay_state_abi marker and never JIT-compiles, so this build is required.
# The service compiles its own small kernel (glm53_k101.so) from service/kernels/k_hcfuse.py on first start
# (needs ninja and nvcc, writes service/build/k101); that is not part of this script.
#
# Environment (all optional; defaults are relative to the repo root, the parent of this script's directory):
#   REPO_ROOT            repo root
#   ENGINE_SRC           patched engine (engine/apply_engine.sh)                  [$REPO_ROOT/engine-src]
#   NATIVE_DIR           where exllamav3_ext*.so is written                       [$REPO_ROOT/native]
#   BUILD_DIR            object files and ninja state (reused by a re-run)         [$NATIVE_DIR/.build]
#   PYTHON               venv interpreter with torch and setuptools               [$REPO_ROOT/.venv/bin/python, else python3]
#   CUDA_HOME            CUDA toolkit with bin/nvcc                               [/usr/local/cuda-13.3, else from nvcc on PATH]
#   MAX_JOBS             parallel compile jobs                                    [4]
#   TORCH_CUDA_ARCH_LIST target architecture                                      [8.0]
#   NICE                 nice level, 0 disables                                   [19]
#   TMPDIR, TORCH_EXTENSIONS_DIR   scratch locations                              [$BUILD_DIR/tmp, $BUILD_DIR/torch_ext]
#   VERIFY               0 skips the engine.sha256 check before building         [1]
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$HERE/.." && pwd)}
ENGINE_SRC=${ENGINE_SRC:-$REPO_ROOT/engine-src}
NATIVE_DIR=${NATIVE_DIR:-$REPO_ROOT/native}
BUILD_DIR=${BUILD_DIR:-$NATIVE_DIR/.build}
MAX_JOBS=${MAX_JOBS:-4}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
NICE=${NICE:-19}
VERIFY=${VERIFY:-1}
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then PYTHON=$REPO_ROOT/.venv/bin/python; else PYTHON=python3; fi
fi
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -x /usr/local/cuda-13.3/bin/nvcc ]; then CUDA_HOME=/usr/local/cuda-13.3
  elif command -v nvcc >/dev/null 2>&1; then CUDA_HOME=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
  else CUDA_HOME=/usr/local/cuda-13.3; fi
fi
case "$MAX_JOBS" in ''|*[!0-9]*|0) echo "build_native: MAX_JOBS must be a positive integer" >&2; exit 2 ;; esac

die() { echo "build_native: $*" >&2; exit 1; }

[ -f "$ENGINE_SRC/setup.py" ] && [ -d "$ENGINE_SRC/exllamav3/exllamav3_ext" ] || die "no patched engine at $ENGINE_SRC (run engine/apply_engine.sh first)"
grep -q glm_replay_state_abi "$ENGINE_SRC/exllamav3/exllamav3_ext/bindings.cpp" || die "$ENGINE_SRC is not the patched engine (no glm_replay_state_abi in bindings.cpp)"
[ -x "$CUDA_HOME/bin/nvcc" ] || die "no nvcc at $CUDA_HOME/bin/nvcc (set CUDA_HOME)"
command -v "$PYTHON" >/dev/null 2>&1 || die "no interpreter: $PYTHON"

if [ "$VERIFY" != 0 ]; then
  "$HERE/apply_engine.sh" --verify
fi

echo "python : $("$PYTHON" -c 'import sys; print(sys.version.split()[0], sys.executable)')"
TORCH_INFO=$("$PYTHON" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null) || die "$PYTHON cannot import torch (run scripts/setup_venv.sh)"
echo "torch  : $TORCH_INFO"
echo "nvcc   : $("$CUDA_HOME/bin/nvcc" --version | sed -n 's/^Cuda compilation tools, //p')"
case "$TORCH_INFO" in 2.14.1+cu130*) ;; *) echo "warning: the measured setup used torch 2.14.1+cu130" >&2 ;; esac
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 13\.' || echo "warning: the measured setup used CUDA toolkit 13.3" >&2

mkdir -p "$NATIVE_DIR" "$BUILD_DIR/tmp" "$BUILD_DIR/torch_ext"
RUN=()
if [ "$NICE" != 0 ]; then
  RUN+=(nice -n "$NICE")
  command -v ionice >/dev/null 2>&1 && RUN+=(ionice -c3)
fi

echo "building into $NATIVE_DIR (MAX_JOBS=$MAX_JOBS, arch $TORCH_CUDA_ARCH_LIST, nice $NICE); log: $BUILD_DIR/build.log"
START=$(date +%s)
# env -u: drop the variables that would change the recipe if they happened to be set in the caller's shell
( cd "$ENGINE_SRC" && env \
    -u EXLLAMA_NOCOMPILE -u EXLLAMA_VERBOSE -u EXLLAMA_EXT_DEBUG -u EXLLAMA_EXT_LINEINFO -u EXLLAMA_EXT_COMPRESS \
    -u PYTORCH_NVCC -u CUDAHOSTCXX -u NVCC_PREPEND_FLAGS -u NVCC_APPEND_FLAGS -u CUDA_PATH \
    CUDA_HOME="$CUDA_HOME" \
    PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH" \
    LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    CUDA_VISIBLE_DEVICES= \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    MAX_JOBS="$MAX_JOBS" \
    PYTHONDONTWRITEBYTECODE=1 \
    TMPDIR="${TMPDIR:-$BUILD_DIR/tmp}" \
    TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$BUILD_DIR/torch_ext}" \
    "${RUN[@]}" "$PYTHON" setup.py build_ext --build-lib "$NATIVE_DIR" --build-temp "$BUILD_DIR" \
) > "$BUILD_DIR/build.log" 2>&1 || { tail -n 40 "$BUILD_DIR/build.log" >&2; die "build failed, see $BUILD_DIR/build.log"; }
SECS=$(( $(date +%s) - START ))

shopt -s nullglob
SOS=("$NATIVE_DIR"/exllamav3_ext*.so)
[ "${#SOS[@]}" = 1 ] || die "expected one exllamav3_ext*.so in $NATIVE_DIR, found ${#SOS[@]}"
SO=${SOS[0]}
if command -v sha256sum >/dev/null 2>&1; then SUM=$(sha256sum "$SO" | cut -d' ' -f1); else SUM=$(shasum -a 256 "$SO" | cut -d' ' -f1); fi
grep -aq glm_replay_state_abi "$SO" || die "$SO lacks the glm_replay_state_abi marker"
echo "built  : $SO"
echo "sha256 : $SUM"
echo "wall   : ${SECS}s"
if command -v readelf >/dev/null 2>&1; then
  echo "needed : $(readelf -d "$SO" | awk '/NEEDED/{gsub(/[\[\]]/,"",$5); printf "%s ", $5}')"
fi
echo "A rebuild is not byte-identical to the measured setup's extension (object files embed source and build paths);"
echo "compare dynamic symbols if you need to: nm -D --defined-only <so> | c++filt | sort"
echo "Not loaded. Smoke test (CPU only):"
echo "  CUDA_VISIBLE_DEVICES= PYTHONPATH=$NATIVE_DIR:$ENGINE_SRC $PYTHON -c 'import exllamav3_ext as e; print(e.glm_replay_state_abi)'"
