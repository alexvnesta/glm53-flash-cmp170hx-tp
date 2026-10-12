#!/usr/bin/env bash
# Build <repo>/engine-src: turboderp's ExLlamaV3 at the pinned commit (UPSTREAM.txt) plus the patches in patches/,
# then check that every file is byte-identical to the engine this setup was measured with (engine.sha256) and that
# the engine files the GLM service hash-pins at startup match.
#
#   engine/apply_engine.sh            clone, patch, verify
#   engine/apply_engine.sh --verify   verify an existing engine-src only (also useful after build_native.sh)
#
# Environment (all optional; defaults are relative to the repo root, the parent of this script's directory):
#   REPO_ROOT    repo root
#   ENGINE_SRC   target directory, must not exist for a build            [$REPO_ROOT/engine-src]
#   UPSTREAM_URL where to fetch ExLlamaV3 from (URL or local clone)      [upstream_url in UPSTREAM.txt]
#   PATCH_DIR    patch series, applied in file-name order                [engine/patches]
#   ENGINE_SHA   expected sha256 list                                    [engine/engine.sha256]
#   SERVICE_GLM  service/glm directory holding the pin tables            [$REPO_ROOT/service/glm]
#   PYTHON       interpreter for the pin check                           [$REPO_ROOT/.venv/bin/python, else python3]
# Needs git and sha256sum (or shasum). Nothing is installed or written outside ENGINE_SRC.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$HERE/.." && pwd)}
ENGINE_SRC=${ENGINE_SRC:-$REPO_ROOT/engine-src}
PATCH_DIR=${PATCH_DIR:-$HERE/patches}
ENGINE_SHA=${ENGINE_SHA:-$HERE/engine.sha256}
UPSTREAM_FILE=$HERE/UPSTREAM.txt
SERVICE_GLM=${SERVICE_GLM:-$REPO_ROOT/service/glm}
PIN_CHECK=$HERE/check_engine_pins.py
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then PYTHON=$REPO_ROOT/.venv/bin/python; else PYTHON=python3; fi
fi

MODE=build
case "${1:-}" in
  "") ;;
  --verify) MODE=verify ;;
  -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
  *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
esac

die() { echo "apply_engine: $*" >&2; exit 1; }
key() { sed -n "s/^$1:[[:space:]]*//p" "$UPSTREAM_FILE" | head -n1 | sed 's/[[:space:]]*$//'; }

BASE_COMMIT=$(key base_commit)          # pinned: the patches only apply to this commit
UPSTREAM_URL=${UPSTREAM_URL:-$(key upstream_url)}
[ -n "$BASE_COMMIT" ] && [ -n "$UPSTREAM_URL" ] || die "UPSTREAM.txt needs upstream_url and base_commit"
[ -f "$ENGINE_SHA" ] || die "missing $ENGINE_SHA"

if command -v sha256sum >/dev/null 2>&1; then SHA256=(sha256sum); else SHA256=(shasum -a 256); fi

verify_tree() {
  [ -d "$ENGINE_SRC" ] || die "$ENGINE_SRC does not exist"
  local total ok tmp
  tmp=$(mktemp -d)
  total=$(wc -l < "$ENGINE_SHA" | tr -d ' ')
  ( cd "$ENGINE_SRC" && "${SHA256[@]}" -c "$ENGINE_SHA" ) > "$tmp/check.txt" 2>&1 || true
  ok=$(grep -c ': OK$' "$tmp/check.txt" || true)
  # same file set: nothing missing, nothing extra (__pycache__ from running the code is ignored)
  ( cd "$ENGINE_SRC" && find . \( -path ./.git -o -name __pycache__ \) -prune -o \( -type f -o -type l \) -print | sed 's|^\./||' | LC_ALL=C sort ) > "$tmp/have.txt"
  cut -c67- "$ENGINE_SHA" | LC_ALL=C sort > "$tmp/want.txt"
  if [ "$ok" = "$total" ] && cmp -s "$tmp/have.txt" "$tmp/want.txt"; then
    echo "verify: $ok/$total files byte-identical to engine.sha256, file set exact"
    rm -rf "$tmp"
    return 0
  fi
  echo "verify: FAILED, $ok/$total files match" >&2
  grep -v ': OK$' "$tmp/check.txt" | head -20 >&2 || true
  diff "$tmp/want.txt" "$tmp/have.txt" | head -20 >&2 || true
  rm -rf "$tmp"
  return 1
}

check_pins() {
  if [ -d "$SERVICE_GLM" ]; then
    "$PYTHON" -I "$PIN_CHECK" "$ENGINE_SRC" "$SERVICE_GLM"
  else
    echo "pins: $SERVICE_GLM not found, service pin check skipped (run check_engine_pins.py ENGINE_SRC SERVICE_GLM later)"
  fi
}

if [ "$MODE" = verify ]; then
  verify_tree
  check_pins
  exit 0
fi

if [ -e "$ENGINE_SRC" ] && [ -n "$(ls -A "$ENGINE_SRC" 2>/dev/null)" ]; then
  die "$ENGINE_SRC already exists. Remove it to rebuild, or run with --verify to check it."
fi
ls "$PATCH_DIR"/*.patch >/dev/null 2>&1 || die "no patches in $PATCH_DIR"

echo "fetching $UPSTREAM_URL @ $BASE_COMMIT"
mkdir -p "$ENGINE_SRC"
git -C "$ENGINE_SRC" init -q
# no line-ending or whitespace rewriting, whatever the user's git config says
git -C "$ENGINE_SRC" config core.autocrlf false
git -C "$ENGINE_SRC" config advice.detachedHead false
git -C "$ENGINE_SRC" remote add origin "$UPSTREAM_URL"
if ! git -C "$ENGINE_SRC" fetch -q --depth 1 origin "$BASE_COMMIT" 2>/dev/null; then
  echo "fetching the commit directly failed, fetching full history"
  git -C "$ENGINE_SRC" fetch -q origin
fi
git -C "$ENGINE_SRC" checkout -q --detach "$BASE_COMMIT"
[ "$(git -C "$ENGINE_SRC" rev-parse HEAD)" = "$BASE_COMMIT" ] || die "checked out the wrong commit"
[ -z "$(git -C "$ENGINE_SRC" status --porcelain)" ] || die "fresh checkout is not clean"

for patch in "$PATCH_DIR"/*.patch; do
  git -C "$ENGINE_SRC" apply --check --whitespace=nowarn "$patch" || die "$(basename "$patch") does not apply"
  git -C "$ENGINE_SRC" apply --whitespace=nowarn "$patch"
  echo "applied $(basename "$patch")"
done

verify_tree
check_pins
echo "engine-src ready: $ENGINE_SRC"
echo "next: engine/build_native.sh, then PYTHONPATH=\$REPO_ROOT/native:\$REPO_ROOT/engine-src:\$REPO_ROOT/service/glm"
