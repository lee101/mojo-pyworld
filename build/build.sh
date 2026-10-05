#!/usr/bin/env bash
# Build the mojo-pyworld kernels into a shared library.
#
# MOJO_NOTES.md section 0: MODULAR_HOME must point at the env's share/max or
# every import fails with "unable to locate module 'std'". `pixi run` sets it,
# but this script is also runnable directly, so set it defensively.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/src/world.mojo"
OUT="${ROOT}/build/libworld.so"

if [[ -z "${MODULAR_HOME:-}" ]]; then
    MOJO_BIN="$(command -v mojo)"
    export MODULAR_HOME="$(dirname "$(dirname "${MOJO_BIN}")")/share/max"
fi

mkdir -p "${ROOT}/build"

# -o takes an output FILE path, not a directory (MOJO_NOTES.md section 0).
mojo build --emit shared-lib -o "${OUT}" "${SRC}"

echo "built ${OUT}"