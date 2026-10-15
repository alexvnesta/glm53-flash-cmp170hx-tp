#!/usr/bin/env bash
# Explicit Weightless variant; requires a separate prepared engine/native build.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GLM_WEIGHTLESS_VECTOR:?Set the path to the authorized pinned GLP-44 GGUF file}"
export GLM_WEIGHTLESS_SHA256="${GLM_WEIGHTLESS_SHA256:-0ccce6b748f87da81505ff2bc3ca82429110940b6c0def60063d304785ccc00c}"
export GLM_WEIGHTLESS_BASE_REVISION="${GLM_WEIGHTLESS_BASE_REVISION:-3f1971b7b5f7a528c9c4ef6212c8785298a8c24a}"
export GLM_WEIGHTLESS_ALPHA="${GLM_WEIGHTLESS_ALPHA:-2.0}"
export GLM_WEIGHTLESS_VALIDATE="${GLM_WEIGHTLESS_VALIDATE:-1}"
export GLM_WEIGHTLESS_RECEIPTS="${GLM_WEIGHTLESS_RECEIPTS:-$ROOT/weightless-receipts}"
export GLM53_ENGINE_SRC="${GLM53_ENGINE_SRC:-$ROOT/engine-weightless}"
export GLM53_NATIVE_DIR="${GLM53_NATIVE_DIR:-$ROOT/native-weightless}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/.cache/weightless-triton}"
exec "$ROOT/scripts/run_tp.sh" "$@"
