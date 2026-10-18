#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# First argument: strategy (default: controller)
# Remaining arguments: supported Python runner options.
STRATEGY="${1:-controller}"

if (( $# > 0 )); then
    shift
fi

exec bash "$HERE/run_experiment.sh" "e6b" "$STRATEGY" "$@"
