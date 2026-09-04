#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_VERSION="${JULIA_VERSION-+1.11}"

run_validator() {
    local validator="$1"
    if [[ -n "$JULIA_VERSION" ]]; then
        "$JULIA_BIN" "$JULIA_VERSION" --startup-file=no \
            --project="$REPOSITORY_ROOT" "$validator"
    else
        "$JULIA_BIN" --startup-file=no \
            --project="$REPOSITORY_ROOT" "$validator"
    fi
}

status=0
if ! run_validator "$SCRIPT_DIR/validate_gep.jl"; then
    status=1
fi
if ! run_validator "$SCRIPT_DIR/validate_msuc.jl"; then
    status=1
fi

exit "$status"
