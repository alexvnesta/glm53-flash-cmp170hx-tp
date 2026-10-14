#!/usr/bin/env bash
# Compatibility entry point: use the qualified tensor-parallel recipe.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$ROOT/scripts/run_tp.sh" "$@"
