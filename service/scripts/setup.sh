#!/usr/bin/env bash
# Compatibility entry point for the pinned repository setup.
# Downloads and model conversion remain a separate explicit get_models.sh step.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$ROOT/scripts/setup_venv.sh"
"$ROOT/engine/apply_engine.sh"
"$ROOT/engine/build_native.sh"
echo "Next: $ROOT/scripts/get_models.sh, then $ROOT/scripts/run_tp.sh"
