#!/usr/bin/env bash
# Prepare and verify a fresh source profile, then compile it without loading CUDA.
set -euo pipefail
PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(cd "$PROFILE_DIR/../.." && pwd)"
: "${1:?Usage: build.sh OUTPUT_DIRECTORY}"
PYTHON="${PYTHON:-$RECIPE_ROOT/.venv/bin/python}"
"$PYTHON" "$PROFILE_DIR/prepare.py" "$1"
PREPARED="$(cd "$1" && pwd)"
# prepare.py has just checked this variant's full manifest and startup pins;
# the general builder's VERIFY=1 would instead check the unmodified base tree.
ENGINE_SRC="$PREPARED/engine" NATIVE_DIR="$PREPARED/native" VERIFY=0 PYTHON="$PYTHON" \
  "$RECIPE_ROOT/engine/build_native.sh"
