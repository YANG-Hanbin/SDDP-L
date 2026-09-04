#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT/experiments/normalized_relu"
JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_VERSION="${JULIA_VERSION-+1.11}"

run_julia() {
    local script_path="$1"
    if [[ -n "$JULIA_VERSION" ]]; then
        "$JULIA_BIN" "$JULIA_VERSION" --startup-file=no --project=. "$script_path"
    else
        "$JULIA_BIN" --startup-file=no --project=. "$script_path"
    fi
}

cd "$ROOT"

batch_status=0

echo "[$(date)] Starting GEP SDDP-L-B NormalizedReLUC batch"
if ! run_julia "$SCRIPT_DIR/run_gep.jl"; then
    echo "[$(date)] GEP batch failed" >&2
    batch_status=1
fi

echo "[$(date)] Starting MSUC SDDP-L-B NormalizedReLUC batch"
if ! run_julia "$SCRIPT_DIR/run_msuc.jl"; then
    echo "[$(date)] MSUC batch failed" >&2
    batch_status=1
fi

if ((batch_status != 0)); then
    echo "[$(date)] SDDP-L-B NormalizedReLUC batch completed with failures" >&2
    exit "$batch_status"
fi

echo "[$(date)] Completed SDDP-L-B NormalizedReLUC batch successfully"
