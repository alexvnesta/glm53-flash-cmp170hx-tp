#!/usr/bin/env bash
# Explicit four-slot GPU-resident profile; the general launcher keeps its defaults.
set -euo pipefail
PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(cd "$PROFILE_DIR/../.." && pwd)"
: "${GLM53_COMMUNITY_PROFILE:?Set GLM53_COMMUNITY_PROFILE to the prepared directory}"
: "${GLM_WEIGHTLESS_VECTOR:?Set GLM_WEIGHTLESS_VECTOR to your authorized GLP-44 alpha2 vector}"
export GLM53_ENGINE_SRC="$GLM53_COMMUNITY_PROFILE/engine"
export GLM53_NATIVE_DIR="$GLM53_COMMUNITY_PROFILE/native"
export GLM53_SERVICE_DIR="$GLM53_COMMUNITY_PROFILE/service/glm"
export RECURRENT_STATE_DTYPE=fp16
export GLM_WEIGHTLESS_ALPHA=2.0 GLM_WEIGHTLESS_VALIDATE=1
export GLM_WEIGHTLESS_SHA256=0ccce6b748f87da81505ff2bc3ca82429110940b6c0def60063d304785ccc00c
export GLM_WEIGHTLESS_BASE_REVISION=3f1971b7b5f7a528c9c4ef6212c8785298a8c24a
export GLM_WEIGHTLESS_RECEIPTS="$GLM53_COMMUNITY_PROFILE/receipts"
exec "$RECIPE_ROOT/scripts/run_tp.sh" -ambs 4 --concurrent-max-sequences 4 --experimental-json-mode "$@"
