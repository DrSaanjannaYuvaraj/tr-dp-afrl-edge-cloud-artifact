#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# First argument: strategy (default: tr_dp_afrl)
# Remaining arguments: supported Python runner options.
STRATEGY="${1:-tr_dp_afrl}"

if (( $# > 0 )); then
    shift
fi

exec bash "$HERE/run_experiment.sh" "e2" "$STRATEGY" "$@"
